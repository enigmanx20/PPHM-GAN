import os
import time, datetime, copy, math
import pprint
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

import dataset_utils
from submit_utils import  creat_project_dir, populate_project_dir
from losses import *
from logger import  TFLogger, Logger
from models import Generator, Discriminator
import utils
from config import config

start_time = time.time()
timestamp = time.strftime("%d%b%Y_%H%M%SUTC", time.gmtime())
project_name = timestamp
project_dir, log_dir, sample_dir, snapshot_dir = creat_project_dir(config['results_dir'], project_name)
populate_project_dir(project_dir)
print(f'Save project in .. {project_dir}')

logger = TFLogger(log_dir)
txtlogger = Logger(file_name=os.path.join(project_dir, "log.txt"), file_mode="w", should_flush=True)
pprint.pprint(config)

torch.backends.cudnn.benchmark = True
utils.seed_rng(config['seed'])

n_critics = config['n_critics']
minibatch = config['minibatch']

image_size = config['image_size']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1 and config['multi-gpus']:
    is_parallel = True
else:
    is_parallel = False

# prepair dataloaders
loaders = dataset_utils.get_data_loaders(**{**config, 'batch_size': minibatch})
loader = loaders[0]
val_loader = loaders[1]

# prepair models
G = Generator(**config)
D = Discriminator(**config)

Gs = copy.deepcopy(G)
Ds = copy.deepcopy(D)

if config['restart']>0:
    print('Load checkpoints...')
    sd_G = utils.strip_sd(torch.load('./snapshots/{:0=5}-G.ckpt'.format(config['restart']), 
                                                map_location=torch.device('cpu')))
    sd_D = utils.strip_sd(torch.load('./snapshots/{:0=5}-D.ckpt'.format(config['restart']), 
                                                map_location=torch.device('cpu')))
    G.load_state_dict(sd_G, strict=True)
    D.load_state_dict(sd_D, strict=True)
    
    Gs.load_state_dict(torch.load('./snapshots/{:0=5}-Gs.ckpt'.format(config['restart'])), strict=True)
    Ds.load_state_dict(torch.load('./snapshots/{:0=5}-Ds.ckpt'.format(config['restart'])), strict=True)
utils.print_network(G)
utils.print_network(D)
if is_parallel:
    config['sample_itr'] //= torch.cuda.device_count()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
G.to(device)
D.to(device)

# prepair optimizer and loss criteria
g_optimizer = Adam(G.parameters(), lr=config['lr'])
d_optimizer = Adam(D.parameters(), lr=config['lr']* config['n_critics'])
clip_norm = config['clip_norm']
cls_crite = nn.CrossEntropyLoss()
recon_crite = utils.AbsLoss()
g_scaler = GradScaler()
d_scaler = GradScaler()

if config['restart']>0:
    ckpt = torch.load('./snapshots/{:0=5}-ckpt.ckpt'.format(config['restart']))
    g_optimizer.load_state_dict(ckpt['gop'])
    d_optimizer.load_state_dict(ckpt['dop'])
    g_scaler.load_state_dict(ckpt['gsl'])   
    d_scaler.load_state_dict(ckpt['dsl'])   

# prepair fixed images and lables for sampling
x_fixed, c_fixed = next(iter(val_loader))
x_fixed_cpu = x_fixed.to(torch.float32)
x_fixed = x_fixed.to(device, torch.float32)
cls_fixed = torch.ones([minibatch, 1], dtype=torch.long).to(device)

# prepair label tensors
cls_onehot = torch.FloatTensor(minibatch, config['c_dim']).to(device)
cls_onehot.zero_()
y_ = utils.Distribution(torch.zeros(minibatch, requires_grad=False))
y_.init_distribution('categorical', num_categories=4)
y_.sample_()

# prologue of the training loop
itr = config['start_itr']
kimg = config['restart'] if config['restart'] >0 else config['start_kimg']
epoch = config['start_epoch']
stats = {'itr': itr, 'logger':logger, 'kimg':kimg, 'epochs': epoch, 'device': device}
loader.sampler.start_itr = itr

@torch.no_grad()
def sample(G, is_training, x_fixed, cls_onehot, cls_fixed, kimg, sample_dir, config, model_name):
    G.training = is_training
    x_fake_list = [x_fixed] 
    for i in range(config['c_dim']):
        cls_onehot.zero_()
        cls_onehot.scatter_(1, cls_fixed * i, 1)
        x_fake_list.append(G(x_fixed, cls_onehot))
    x_concat = torch.cat(x_fake_list, dim=3)
    sample_path = os.path.join(sample_dir, '{}-{:0=5}-images.jpg'.format(model_name, int(kimg)))
    save_image(utils.denorm(x_concat.data.cpu()), 
                             sample_path, nrow=1, padding=0)
    print('Saved real and fake images into {}...'.format(sample_path))

sample(G, True, x_fixed, cls_onehot, cls_fixed, kimg, sample_dir, config, 'G_train')

while True:
    print ('itr: {itr}, kimg: {kimg}, epochs: {epochs}'.format(**stats) )
    stats.update({'epochs': epoch})
    pbar = tqdm(loader)
    
    for i, (reals, cls) in enumerate(pbar):
        item = {}
        stats.update({'itr': itr})
        lr = config['lr']
        if config['cos_lr'] and kimg>config['cos_kimg']:
            lr *= 0.5 * (1. + math.cos(math.pi * (kimg - config['restart']) / config['cos_kimg']) )
        G_lr = lr
        D_lr = lr * config['n_critics']
        
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = G_lr
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = D_lr
        item['misc/G_lr'] = G_lr
        item['misc/D_lr'] = D_lr    
        
        batch_size = reals.size(0)
        num_real = batch_size//2
        
        reals = reals.to(device)
        cls = cls.view([-1, 1]).to(device, torch.long)
        cls_onehot.zero_()
        cls_onehot.scatter_(1, cls, 1)
        
        y_.sample_()
        cls_random = y_.clone().view([-1, 1]).to(device, torch.long)
        y = cls_onehot.clone().detach()
        y.zero_()
        y.scatter_(1, cls_random, 1)
        
        # train D
        G.train()
        fake = G(reals, y)
        D_out =D(torch.cat([reals, fake], dim=0))  
        D_real, D_fake = torch.split(D_out, batch_size)
        
        if 'simplegp' in config['d_loss']:
            D_loss = globals()[config['d_loss']](D_fake[:,0], D_real[:,0], D, fake, reals, **stats)
        else:
            D_loss = globals()[config['d_loss']](D_fake[:,0], D_real[:,0], **stats)
            
        G_loss = globals()[config['g_loss']](D_fake[:, 0], **stats)
            
        if config['lambda_cls'] > 0.0:   
            cls_loss_d = cls_crite(D_real[:, 1:], cls.squeeze())
            cls_loss_g = cls_crite(D_fake[:, 1:], cls_random.squeeze())
            D_loss += config['lambda_cls'] * cls_loss_d
            G_loss += config['lambda_cls'] * cls_loss_g    

            item['D/classification_loss'] = cls_loss_d.item()
            item['G/classification_loss'] = cls_loss_g.item()

        d_optimizer.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True) 
        if config['SN'] and config['OR']:
            utils.ortho(D, skip_module=[])
        d_scaler.unscale_(d_optimizer)
        nn.utils.clip_grad_norm_(D.parameters(), clip_norm)
        d_scaler.step(d_optimizer)
        d_scaler.update()
        
        #train G
        if config['lambda_recon'] > 0.0:
            refake =  G(fake.detach(), cls_onehot, **stats)
            D_refake = D(refake)
            recon_loss = recon_crite(reals, refake)
            G_loss += config['lambda_recon'] * recon_loss
            
            item['G/reconstruction_loss'] = recon_loss.item()
            
        g_optimizer.zero_grad() 
        g_scaler.scale(G_loss).backward(retain_graph=False) 
        if config['SN'] and config['OR']:
            utils.ortho(G, skip_module=[])
        g_scaler.unscale_(g_optimizer)
        nn.utils.clip_grad_norm_(G.parameters(), clip_norm)
        g_scaler.step(g_optimizer)
        g_scaler.update()
        
        utils.ema(Gs, G, is_parallel=is_parallel, beta=0.99, beta_nontrainable=1.0)
        utils.ema(Ds, G, is_parallel=is_parallel, beta=0.99, beta_nontrainable=1.0)

        utils.lerp_buffer(Gs, G, is_parallel=is_parallel, beta=1.0)
        
        # epilogue
        itr += 1
        kimg += batch_size /1000
        
        if (itr)%5 == 0:
            item['misc/ite'] = itr
            item['misc/epochs'] = stats['epochs']
            item['misc/kimg'] = kimg
            item['misc/max_memory_allocated'] = (torch.cuda.max_memory_allocated(device=device) / 2**30)
            item['misc/max_memory_cached'] = (torch.cuda.max_memory_reserved(device=device) / 2**30)
            stats.update({'kimg': kimg})
            for tag, value in item.items():
                logger.scalar_summary(tag, value, itr+1)
          
        if (itr)%50 == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            print ("Elapsed [{}], itr: {:2f}, D_loss: {:3f}, G_loss: {:3f}, kimg: {:2f}".format(et, itr,
                                               D_loss.item(), G_loss.item(), int(kimg)) )
            
        if (itr)%config['log_itr'] == 0:
            utils.log_sv(G, **stats)
            utils.log_sv(D, **stats)
            
        if (itr)%config['snapshot_itr'] == 0:     
            G_path = os.path.join(snapshot_dir, '{:0=5}-G.ckpt'.format(int(kimg)))
            D_path = os.path.join(snapshot_dir, '{:0=5}-D.ckpt'.format(int(kimg)))
            torch.save(G.state_dict(), G_path)
            torch.save(D.state_dict(), D_path)        

            Gs_path = os.path.join(snapshot_dir, '{:0=5}-Gs.ckpt'.format(int(kimg)))
            Ds_path = os.path.join(snapshot_dir, '{:0=5}-Ds.ckpt'.format(int(kimg)))
            torch.save(Gs.state_dict(), Gs_path)
            torch.save(Ds.state_dict(), Ds_path)

            ckpt_path = os.path.join(snapshot_dir, '{:0=5}-ckpt.ckpt'.format(int(kimg)))
            torch.save({
                'dop':d_optimizer.state_dict(),
                'gop':g_optimizer.state_dict(),
                'dsl':d_scaler.state_dict(),
                'gsl':d_scaler.state_dict(),
             }, ckpt_path)
            print('Saved model checkpoints into {}...'.format(snapshot_dir))
        
        if (itr)%config['sample_itr'] == 0:
            print ('itr: {itr}, kimg: {kimg}'.format(**stats) )
            sample(G, True, x_fixed, cls_onehot, cls_fixed, kimg, sample_dir, config, 'G_train')
            sample(Gs, False, x_fixed.cpu(), cls_onehot.cpu(), cls_fixed.cpu(), kimg, sample_dir, config, 'Gs_eval')
        if kimg > config['stop_kimg']:
            break
    epoch += 1
    if kimg > config['stop_kimg']:
            break
    
        
                   
txtlogger.close()
        
        
        
        
        
        
        
        

