import numpy as np
import torch
from utils import random_horizontal_flip
from model import *

def train(x,c, config):
    
    '''
    x : data
    c : condition
    config : configuration from main.py
    '''
    
    lambda_recon = config.lambda_recon
    lambda_match_zc = config.lambda_match_zc
    lambda_translation = config.lambda_translation
    lambda_match_xcc = config.lambda_match_xcc
    lambda_cycle = config.lambda_cycle
    lambda_transport = config.lambda_transport
    lambda_gp = config.lambda_gp
    lambda_label_pred = config.lambda_label_pred
    
    model_dir = config.model_dir
    
    batch_size = config.batch_size
    n_iter = config.n_iter
    iter_critic = config.iter_critic
    print_period = config.print_period
    iter_critic = config.iter_critic
    init_lr = config.init_lr
    lr_update_period = config.lr_update_period
    lr_decay_start_iter = config.lr_decay_start_iter
           
    zdim = config.zdim

    # Construct our model by instantiating the class defined above
    encoder = Encoder(zdim=zdim, cdim = np.shape(c)[1])
    decoder = Decoder(zdim=zdim, cdim = np.shape(c)[1])
    translator = Translator(zdim=zdim, cdim = np.shape(c)[1])
    discriminator_zc = Discriminator_zc(zdim=zdim, cdim = np.shape(c)[1])
    discriminator_xcc = Discriminator_xcc(cdim = np.shape(c)[1])

    encoder.cuda()
    decoder.cuda()
    translator.cuda()
    discriminator_zc.cuda()
    discriminator_xcc.cuda()

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the nn.Linear
    # module which is members of the model.
    mse_criterion = torch.nn.MSELoss()
    bce_criterion = torch.nn.BCELoss()
    mse_criterion.cuda()
    bce_criterion.cuda()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=init_lr, betas = [0.5, 0.999])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=init_lr, betas = [0.5, 0.999])
    translator_optimizer = torch.optim.Adam(translator.parameters(), lr=init_lr, betas = [0.5, 0.999])
    discriminator_zc_optimizer = torch.optim.Adam(discriminator_zc.parameters(), lr=init_lr, betas = [0.5, 0.999])
    discriminator_xcc_optimizer = torch.optim.Adam(discriminator_xcc.parameters(), lr=init_lr, betas = [0.5, 0.999])
    
    print('train start')
    
    t0 = time.time()

    for t in range(n_iter):
        # Generate and pre-process batch
        src_batch_idx = np.random.choice(np.shape(x)[0], batch_size)
        tar_batch_idx = np.random.choice(np.shape(x)[0], batch_size)
        x_src_batch, c_src_batch = x[src_batch_idx].cuda(), c[src_batch_idx].cuda()
        x_tar_batch, c_tar_batch = x[tar_batch_idx].cuda(), c[tar_batch_idx].cuda()

        x_src_batch, c_src_batch = random_horizontal_flip(x_src_batch, c_src_batch)
        x_tar_batch, c_tar_batch = random_horizontal_flip(x_tar_batch, c_tar_batch)

        x_src_to_tar = translator(x_src_batch, c_src_batch, c_tar_batch)
        x_tar_to_src = translator(x_tar_batch, c_tar_batch, c_src_batch)

        cdim, h, w = np.shape(c_src_batch)[1], 2, 2
        c_src_batch_box = torch.tensor(torch.unsqueeze(c_src_batch, dim=2), dtype=torch.float32).cuda()
        c_tar_batch_box = torch.tensor(torch.unsqueeze(c_tar_batch, dim=2), dtype=torch.float32).cuda()
        c_src_batch_box = torch.reshape(c_src_batch_box @ torch.ones((np.shape(c_src_batch_box)[0], 1, h*w)).cuda(),
                                        (np.shape(c_src_batch_box)[0], cdim, h, w))
        c_tar_batch_box = torch.reshape(c_tar_batch_box @ torch.ones((np.shape(c_tar_batch_box)[0], 1, h*w)).cuda(),
                                        (np.shape(c_tar_batch_box)[0], cdim, h, w))

        ##############################
        # update discriminators
        ##############################
        encoder.eval()
        decoder.eval()
        translator.eval()

        discriminator_zc.train()   
        discriminator_xcc.train()
        discriminator_zc.zero_grad()
        discriminator_xcc.zero_grad()

        z_src = torch.normal(mean = torch.zeros(batch_size, zdim), std = torch.ones(batch_size, zdim)).cuda()
        z_tar = torch.normal(mean = torch.zeros(batch_size, zdim), std = torch.ones(batch_size, zdim)).cuda()

        z_tilde_src = encoder(x_src_batch, c_src_batch)
        z_tilde_tar = encoder(x_tar_batch, c_tar_batch)

        critic_real_src, c_src_pred = discriminator_xcc(x_src_batch, c_tar_batch_box, c_src_batch_box)
        critic_real_tar, c_tar_pred = discriminator_xcc(x_tar_batch, c_src_batch_box, c_tar_batch_box)
        critic_fake_src, _ = discriminator_xcc(x_tar_to_src.detach(), c_tar_batch_box, c_src_batch_box)
        critic_fake_tar, _ = discriminator_xcc(x_src_to_tar.detach(), c_src_batch_box, c_tar_batch_box)

        meanD_real_xcc = (torch.mean(critic_real_src)+torch.mean(critic_real_tar))/2.0
        meanD_fake_xcc = (torch.mean(critic_fake_src)+torch.mean(critic_fake_tar))/2.0

        label_real, label_fake = torch.full((batch_size,), 1.0).cuda(), torch.full((batch_size,), 0.0).cuda()
        errD_real_zc = (bce_criterion(discriminator_zc(z_src, c_src_batch), label_real)
                        +bce_criterion(discriminator_zc(z_tar, c_tar_batch), label_real))/2.0
        errD_fake_zc = (bce_criterion(discriminator_zc(z_tilde_src, c_src_batch), label_fake)
                        +bce_criterion(discriminator_zc(z_tilde_tar, c_tar_batch), label_fake))/2.0

        alpha = torch.rand(x_src_batch.size(0), 1, 1, 1).cuda()
        x_inter_src = (alpha*x_src_batch.data+(1.-alpha)*x_tar_to_src.data).requires_grad_(True)
        c_tar_var = c_tar_batch_box.data.requires_grad_(True)
        c_src_var = c_src_batch_box.data.requires_grad_(True)
        critic_inter_src, _ = discriminator_xcc(x_inter_src, c_tar_var, c_src_var)
        gp_src = torch.mean((torch.sqrt(gradient_norm(critic_inter_src, x_inter_src)**2
                                        + gradient_norm(critic_inter_src, c_tar_var)**2
                                        + gradient_norm(critic_inter_src, c_src_var)**2)-1)**2)

        alpha = torch.rand(x_tar_batch.size(0), 1, 1, 1).cuda()
        x_inter_tar = (alpha*x_tar_batch.data+(1.-alpha)*x_src_to_tar.data).requires_grad_(True)
        critic_inter_tar, _ = discriminator_xcc(x_inter_tar, c_src_var, c_tar_var)
        gp_tar = torch.mean((torch.sqrt(gradient_norm(critic_inter_tar, x_inter_tar)**2
                                        + gradient_norm(critic_inter_tar, c_src_var)**2
                                        + gradient_norm(critic_inter_tar, c_tar_var)**2)-1))**2

        loss_match_zc = 0.5*(errD_real_zc + errD_fake_zc)
        loss_match_xcc = -meanD_real_xcc + meanD_fake_xcc
        loss_gp = 0.5*(gp_src+gp_tar)
        loss_label_pred = (mse_criterion(c_src_pred, c_src_batch)
                           +mse_criterion(c_tar_pred, c_tar_batch))/2.0

        loss_D = (lambda_match_zc*loss_match_zc + lambda_match_xcc*loss_match_xcc
                  + lambda_gp*loss_gp + lambda_label_pred*loss_label_pred)
        loss_D.backward()

        discriminator_zc_optimizer.step()
        discriminator_xcc_optimizer.step()

        if (t+1) % iter_critic == 0:
            ##############################
            # update translator
            ##############################
            encoder.eval()
            decoder.eval()
            discriminator_zc.eval()
            discriminator_xcc.eval()

            translator.train()
            translator.zero_grad()

            # Forward pass: Compute predicted y by passing x to the model
            x_src_to_tar = translator(x_src_batch, c_src_batch, c_tar_batch)
            x_tar_to_src = translator(x_tar_batch, c_tar_batch, c_src_batch)
            return_x_src = translator(x_src_to_tar, c_tar_batch, c_src_batch)
            return_x_tar = translator(x_tar_to_src, c_src_batch, c_tar_batch)

            # calculate mean discrepancy with learned critic
            critic_fake_src, _ = discriminator_xcc(x_tar_to_src, c_tar_batch_box, c_src_batch_box)
            critic_fake_tar, _ = discriminator_xcc(x_src_to_tar, c_src_batch_box, c_tar_batch_box)
            meanD_fake_xcc = (torch.mean(critic_fake_src)+torch.mean(critic_fake_tar))/2.0

            # prepare alexnet input (224 x 224)
            #x_src_alex = normalize_for_alexnet(F.interpolate(x_src_batch, size=224))
            #x_src_to_tar_alex = normalize_for_alexnet(F.interpolate(x_src_to_tar, size=224))
            #x_tar_alex = normalize_for_alexnet(F.interpolate(x_tar_batch, size=224))
            #x_tar_to_src_alex = normalize_for_alexnet(F.interpolate(x_tar_to_src, size=224))

            # Compute and print loss
            loss_match_xcc = -meanD_fake_xcc
            loss_cycle = (mse_criterion(x_src_batch, return_x_src) + mse_criterion(x_tar_batch, return_x_tar))/2.0
            #loss_transport = (mse_criterion(alexnet(x_src_alex), alexnet(x_src_to_tar_alex))
            #                  +mse_criterion(alexnet(x_tar_alex), alexnet(x_tar_to_src_alex)))/2.0
            loss_transport = (mse_criterion(discriminator_xcc(x_src_batch, c_tar_batch, c_src_batch, feature_extract=True),
                                            discriminator_xcc(x_src_to_tar, c_tar_batch, c_src_batch, feature_extract=True))
                              +mse_criterion(discriminator_xcc(x_tar_batch, c_src_batch, c_tar_batch, feature_extract=True),
                                            discriminator_xcc(x_tar_to_src, c_src_batch, c_tar_batch, feature_extract=True)))/2.0

            loss_T = (lambda_match_xcc*loss_match_xcc + lambda_cycle*loss_cycle + lambda_transport*loss_transport)

            # Zero gradients, perform a backward pass, and update the weights.
            loss_T.backward()

            translator_optimizer.step()

            ##############################
            # update generator
            ##############################
            translator.eval()
            discriminator_zc.eval()
            discriminator_xcc.eval()

            encoder.train()
            decoder.train()

            encoder.zero_grad()
            decoder.zero_grad()

            # Forward pass: Compute predicted y by passing x to the model
            z_tilde_src = encoder(x_src_batch, c_src_batch)
            z_tilde_tar = encoder(x_tar_batch, c_tar_batch)
            recon_x_src = decoder(z_tilde_src, c_src_batch)
            recon_x_tar = decoder(z_tilde_tar, c_tar_batch)
            x_src_to_tar = translator(x_src_batch, c_src_batch, c_tar_batch)
            x_tar_to_src = translator(x_tar_batch, c_tar_batch, c_src_batch)

            # -log trick
            errG_zc = (bce_criterion(discriminator_zc(z_tilde_src, c_src_batch), label_real)
                       +bce_criterion(discriminator_zc(z_tilde_tar, c_tar_batch), label_real))/2.0

            #
            x_gen_src = decoder(z_src, c_src_batch)
            x_gen_tar = decoder(z_tar, c_tar_batch)

            # Compute and print loss
            loss_recon = (mse_criterion(x_src_batch, recon_x_src)
                          + mse_criterion(x_tar_batch, recon_x_tar))/2.0
            loss_match_zc = errG_zc
            loss_translation = (mse_criterion(discriminator_xcc(decoder(z_src, c_tar_batch),
                                                                c_src_batch, c_tar_batch,
                                                                feature_extract=True),
                                              discriminator_xcc(translator(x_gen_src, c_src_batch, c_tar_batch),
                                                                c_src_batch, c_tar_batch,
                                                                feature_extract=True))
                               + mse_criterion(discriminator_xcc(decoder(z_tar, c_src_batch),
                                                                 c_tar_batch, c_src_batch,
                                                                 feature_extract=True),
                                               discriminator_xcc(translator(x_gen_tar, c_tar_batch, c_src_batch),
                                                                 c_tar_batch, c_src_batch,
                                                                 feature_extract=True))/2.0)

            loss_G = lambda_recon*loss_recon + lambda_match_zc*loss_match_zc + lambda_translation*loss_translation

            # Zero gradients, perform a backward pass, and update the weights.
            loss_G.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

        if (((t+1) % lr_update_period == 0) & ((t+1) > lr_decay_start_iter)):
            encoder_optimizer.param_groups[0]['lr'] -= init_lr*lr_update_period/(n_iter-lr_decay_start_iter+1)
            decoder_optimizer.param_groups[0]['lr'] -= init_lr*lr_update_period/(n_iter-lr_decay_start_iter+1)
            translator_optimizer.param_groups[0]['lr'] -= init_lr*lr_update_period/(n_iter-lr_decay_start_iter+1)
            discriminator_zc_optimizer.param_groups[0]['lr'] -= init_lr*lr_update_period/(n_iter-lr_decay_start_iter+1)
            discriminator_xcc_optimizer.param_groups[0]['lr'] -= init_lr*lr_update_period/(n_iter-lr_decay_start_iter+1)
            print('Decayed learning rate: %.6f: ' % encoder_optimizer.param_groups[0]['lr'])

        if (t+1) % print_period == 0:
            t1 = time.time()
            print('%d\tloss_recon: %.4f\tloss_match_zc: %.4f\tloss_translation: %.4f\tloss_match_xcc: %.4f\tloss_cycle: %.4f\tloss_transport: %.4f\tloss_gp: %.4f\tloss_label_pred: %.4f\tloss_D: %.4f\tloss_T: %.4f\tloss_G: %.4f'
                  % (t, loss_recon.item(), loss_match_zc.item(), loss_translation.item(), loss_match_xcc.item(), loss_cycle.item(), loss_transport.item(), loss_gp.item(), loss_label_pred.item(), loss_D.item(), loss_T.item(), loss_G.item()))
            print('cumulated time: %.0f' % (t1-t0))
            
    # save the trained models
    torch.save(encoder.state_dict(), '%s/encoder.pth' % model_dir)
    torch.save(decoder.state_dict(), '%s/decoder.pth' % model_dir)
    torch.save(translator.state_dict(), '%s/translator.pth' % model_dir)
    torch.save(discriminator_zc.state_dict(), '%s/discriminator_zc.pth' % model_dir)
    torch.save(discriminator_xcc.state_dict(), '%s/discriminator_xcc.pth' % model_dir)
