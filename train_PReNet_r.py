import os
import argparse
import paddle
from paddle.io import DataLoader
from visualdl import LogWriter
from DerainDataset import *
from utils import *
from paddle.optimizer.lr import MultiStepDecay as MultiStepLR
from SSIM import SSIM
from networks import *
from logger import Logger


parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainL",help='path to training data')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()


def main():
    # record training
    writer = LogWriter(opt.save_path)
    log = Logger(opt.save_path + '/training.log',level='info')
    log.logger.info('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    log.logger.info("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = PReNet_r(recurrent_iter=opt.recurrent_iter)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Optimizer
    scheduler = MultiStepLR(learning_rate=opt.lr, milestones=opt.milestone, gamma=0.2)  # learning rates
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        log.logger.info('resuming by loading epoch %d' % initial_epoch)
        model.set_state_dict(paddle.load(os.path.join(opt.save_path, 'net_epoch%d.pdparams' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        log.logger.info('learning rate %f' % optimizer.get_lr())

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # training curve
            model.eval()
            out_train, _ = model(input_train)
            out_train = paddle.clip(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            log.logger.info("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## epoch training end

        # log the images
        model.eval()
        out_train, _ = model(input_train)
        out_train = paddle.clip(out_train, 0., 1.)

        im_target = make_grid(target_train, nrow=8)
        im_input = make_grid(input_train, nrow=8)
        im_derain = make_grid(out_train, nrow=8)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)


        # save model
        paddle.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pdparams'))
        if epoch % opt.save_freq == 0:
            paddle.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pdparams' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
