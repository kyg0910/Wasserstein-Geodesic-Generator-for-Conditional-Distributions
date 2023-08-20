import numpy as np
import torch
import matplotlib.pyplot as plt
from model import *

def evaluate(x,c,config):
    
    
    zdim = config.zdim
    model_dir = config.model_dir
    generation_dir = config.generation_dir
    
    encoder = Encoder(zdim=zdim, cdim = np.shape(c)[1]).cuda()
    decoder = Decoder(zdim=zdim, cdim = np.shape(c)[1]).cuda()
    translator = Translator(zdim=zdim, cdim = np.shape(c)[1]).cuda()
    discriminator_zc = Discriminator_zc(zdim=zdim, cdim = np.shape(c)[1]).cuda()
    discriminator_xcc = Discriminator_xcc(cdim = np.shape(c)[1]).cuda()

    encoder.load_state_dict(torch.load('%s/encoder.pth' % model_dir))
    decoder.load_state_dict(torch.load('%s/decoder.pth' % model_dir))
    translator.load_state_dict(torch.load('%s/translator.pth' % model_dir))
    discriminator_zc.load_state_dict(torch.load('%s/discriminator_zc.pth' % model_dir))
    discriminator_xcc.load_state_dict(torch.load('%s/discriminator_xcc.pth' % model_dir))

    
    ######################################################## eval 1 ########################################################
    
    print('< Translation Results on Train Data >')

    aspect = 1.0
    n = 10 # number of rows
    m = 4 # numberof columns
    bottom = 0.1; left=0.05
    top=1.-bottom; right = 1.-left
    fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
    #widthspace, relative to subplot size
    wspace=0.15  # set to zero for no spacing
    hspace=wspace/float(aspect)
    #fix the figure height
    figheight= 15 # inch
    figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp

    f, a = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                        wspace=wspace, hspace=hspace)

    #f, a = plt.subplots(10, 4, figsize = (10, 10))
    #fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    #plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
    #                    wspace=wspace, hspace=hspace)

    encoder.eval()
    decoder.eval()
    translator.eval()

    current_c_src = torch.tensor([0./180.0, 0./90.0])
    current_c_tar = torch.tensor([30./180.0, 0./90.0])

    src_batch_idx = np.random.choice(np.where((c.cpu()[:, 0] == current_c_src[0])
                                              & (c.cpu()[:, 1] == current_c_src[1]))[0], n)
    x_src_batch = x[src_batch_idx].cuda()
    recon_x_src = decoder(encoder(x_src_batch, current_c_src.repeat(n, 1).cuda()),
                          current_c_src.repeat(n, 1).cuda())
    x_src_to_tar = translator(x_src_batch, current_c_src.repeat(n, 1).cuda(), current_c_tar.repeat(n, 1).cuda())
    return_x_src = translator(x_src_to_tar, current_c_tar.repeat(n, 1).cuda(), current_c_src.repeat(n, 1).cuda())

    print('source domain label is [%.0f, %.0f]' % (current_c_src.detach().numpy()[0]*180.0,
                                                   current_c_src.detach().numpy()[1]*90.0))
    print('target domain label is [%.0f, %.0f]' % (current_c_tar.detach().numpy()[0]*180.0,
                                                   current_c_tar.detach().numpy()[1]*90.0))
    for i in range(n):
        current_x_src = x_src_batch[i].cpu().detach().numpy()
        current_recon_x_src = recon_x_src[i].cpu().detach().numpy()
        current_x_src_to_tar = x_src_to_tar[i].cpu().detach().numpy()
        current_return_x_src = return_x_src[i].cpu().detach().numpy()

        a[i, 0].imshow(np.uint8(255.0*np.repeat(np.transpose(current_x_src, (1, 2, 0)), 3, axis=2)))
        a[i, 1].imshow(np.uint8(255.0*np.repeat(np.transpose(current_recon_x_src, (1, 2, 0)), 3, axis=2)))
        a[i, 2].imshow(np.uint8(255.0*np.repeat(np.transpose(current_x_src_to_tar, (1, 2, 0)), 3, axis=2)))
        a[i, 3].imshow(np.uint8(255.0*np.repeat(np.transpose(current_return_x_src, (1, 2, 0)), 3, axis=2)))
        if i == 0:
            a[i, 0].set_title("Original", fontsize=10)
            a[i, 1].set_title("Reconstruction", fontsize=10)
            a[i, 2].set_title("Translation", fontsize=10)
            a[i, 3].set_title("Cycle", fontsize=10)
        a[i, 0].axis('off')
        a[i, 1].axis('off')
        a[i, 2].axis('off')
        a[i, 3].axis('off')

    f.savefig('%s/figure1.pdf' % generation_dir)
    
    ######################################################## eval 2 ########################################################  

    print('< Translation Results on Train Data (red frame : real data, blue frame: translated data) >')

    c1_list = [-130.0, -120.0, -110.0, -95.0, -80.0, -70.0, -60.0, -50.0, -35.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0,
              5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 60.0, 70.0, 85.0, 95.0, 110.0, 120.0, 130.0]
    c2_list = [90.0, 65.0, 45.0, 20.0, 15.0, 10.0, 0.0, -10.0, -20.0, -35.0, -40.0]

    aspect = 1.0
    n = len(c2_list) # number of rows
    m = len(c1_list) # numberof columns
    bottom = 0.1; left=0.05
    top=1.-bottom; right = 1.-left
    fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
    #widthspace, relative to subplot size
    wspace=0.15  # set to zero for no spacing
    hspace=wspace/float(aspect)
    #fix the figure height
    figheight= 20 # inch
    figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp

    f, a = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                        wspace=wspace, hspace=hspace)

    encoder.eval()
    decoder.eval()
    translator.eval()

    current_c_src = torch.tensor([0./180.0, 0./90.0])
    src_batch_idx = np.random.choice(np.where((c.cpu()[:, 0] == current_c_src[0])
                                              & (c.cpu()[:, 1] == current_c_src[1]))[0], 1)
    for i in range(len(c1_list)):
        for j in range(len(c2_list)):
            current_c_tar = torch.tensor([c1_list[i]/180.0, c2_list[j]/90.0])
            current_x_src_to_tar = translator(x[src_batch_idx], current_c_src.repeat(1, 1).cuda(),
                                              current_c_tar.repeat(1, 1).cuda()).cpu().detach().numpy()

            a[j, i].set_title('(%.0f, %.0f)' % (c1_list[i], c2_list[j]), fontsize=10)
            a[j, i].imshow(np.uint8(255.0*np.repeat(np.transpose(current_x_src_to_tar[0], (1, 2, 0)), 3, axis=2)))
            a[j, i].set_xticks([])
            a[j, i].set_yticks([])
            plt.setp(a[j, i].spines.values(), color = 'blue', lw = 2)

            if ((c1_list[i] == current_c_src[0]) & (c2_list[j] == current_c_src[1])):
                a[j, i].imshow(np.uint8(255.0*np.repeat(np.transpose(x[src_batch_idx][0].cpu(), (1, 2, 0)), 3, axis=2)))
                plt.setp(a[j, i].spines.values(), color = 'red', lw = 2)
            #a[j, i].axis('off')

    f.savefig('%s/figure2.pdf' % generation_dir)