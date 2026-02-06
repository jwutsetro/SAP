import os
import shutil
import time
import wandb
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch, list_data_collate


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        data, target = data.cuda(args.rank, non_blocking=True), target.cuda(args.rank, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            # Gather loss from all ranks
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=True)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=data.size(0) * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=data.size(0))

        if args.rank == 0:
            print(
                f"Epoch {epoch}/{args.max_epochs} {idx}/{len(loader)} "
                f"loss: {run_loss.avg:.4f} time {time.time() - start_time:.2f}s"
            )
        start_time = time.time()

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    # Now validation is run on all ranks, so all distributed calls are safe.
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data, target = data.cuda(args.rank, non_blocking=True), target.cuda(args.rank, non_blocking=True)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True, is_valid=True)
                # Aggregate acc from all ranks
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            # Print only from rank 0 for clarity
            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)} "
                    f"acc {avg_acc} time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    if args.rank == 0:
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    # Initialize WandB only on rank 0
    if args.rank == 0:
        print('Initializing WandB on rank 0')
        wandb.init(project='Your_Project_Name')
        wandb.config.update({"split": args.split, "training_mode": args.training_mode})
    args.logdir = os.path.join(args.logdir, 'cross_validation_large', str(args.split), args.training_mode)
    os.makedirs(args.logdir, exist_ok=True)

    # Ensure consistent batch sizes if distributed
    if args.distributed and hasattr(train_loader.sampler, 'drop_last'):
        train_loader.sampler.drop_last = True

    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0

    for epoch in range(start_epoch, args.max_epochs):
        if args.rank == 0:
            print(f"\nStarting Epoch {epoch + 1}")

        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if args.rank == 0:
            print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )

        if args.rank == 0:
            print(
                f"Final training  {epoch}/{args.max_epochs - 1} "
                f"loss: {train_loss:.4f} time {time.time() - epoch_time:.2f}s"
            )
            wandb.log({'loss': train_loss, 'epoch': epoch})

        b_new_best = False


        # Validate if conditions met - now done on all ranks
        if (epoch + 1) % args.val_every == 0 and val_loader is not None:
            # Barrier to ensure all ranks start validation together
            if args.distributed:
                torch.distributed.barrier()

            if args.rank == 0:
                print("Starting validation on all ranks...")

            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            # val_avg_acc is aggregated from all GPUs in val_epoch
            final_val_acc = np.mean(val_avg_acc)

            # All ranks have the final_val_acc now,
            # but only rank 0 logs and saves
            if args.rank == 0:
                class_names = ['cervical','thoracic','lumbar','femur','pelvis','humerus','shoulders','clavicles','sternum']
                for class_idx, class_name in enumerate(class_names):
                    # If val_avg_acc is a vector over classes, adjust indexing accordingly
                    # Here val_avg_acc is a single average over multiple iterations.
                    # If you have multiple classes metric, adjust the code.
                    print(f'validation_dice_{class_name} is: {final_val_acc}')
                    wandb.log({f'validation_dice_{class_name}': final_val_acc, 'epoch': epoch})

                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

                print(
                    f"Final validation  {epoch}/{args.max_epochs - 1} "
                    f"acc {final_val_acc} time {time.time() - epoch_time:.2f}s"
                )

                if final_val_acc > val_acc_max:
                    print(f"New best ({val_acc_max:.6f} --> {final_val_acc:.6f}). ")
                    val_acc_max = final_val_acc
                    b_new_best = True
                    save_checkpoint(
                        model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                    )

                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                    if b_new_best:
                        print("Copying to model.pt new best model!!!!")
                        shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0:
        print("Training Finished! Best Accuracy: ", val_acc_max)

    return val_acc_max
