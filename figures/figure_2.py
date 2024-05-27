import numpy as np
import matplotlib.pyplot as plt
import tifffile

class Figure_2:
    """Astigmatism summary for high and low NA"""
    def __init__(self,config):
        self.config = config
    def plot(self,prefixes,hw=10):
    
        fig, ax = plt.subplots(2,5,figsize=(6,3))
        
        stack1 = tifffile.imread(self.config['datapath']+prefixes[0]+'.tif')
        stack2 = tifffile.imread(self.config['datapath']+prefixes[1]+'.tif')
        stack3 = tifffile.imread(self.config['datapath']+prefixes[2]+'.tif')
        stack4 = tifffile.imread(self.config['datapath']+prefixes[3]+'.tif')
        stacks = [stack1,stack2]
        pos = [(66,43),(58,73)]
        
        for n in range(2):
            x0,y0 = pos[n]
            ax[n,0].imshow(stacks[n][0,x0-hw:x0+hw,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,1].imshow(stacks[n][10,x0-hw:x0+hw,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,2].imshow(stacks[n][-1,x0-hw:x0+hw,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,3].imshow(stacks[n][:,x0,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,4].imshow(stacks[n][:,x0-hw:x0+hw,y0],cmap='coolwarm')
            
            ticks = [0,2]
            for m in range(5):
                ax[n,m].set_xticks(np.linspace(0,19,len(ticks)))
                ax[n,m].set_yticks(np.linspace(0,19,len(ticks)))
                ax[n,m].set_xticklabels(ticks); ax[n,m].set_yticklabels(ticks)
                ax[n,m].set_xlabel('um',labelpad=0.5)
                if m > 0:
                    ax[n,m].set_ylabel('um',labelpad=0.5)
                else:
                    if n == 0:
                        label='High'
                    else:
                        label='Low'
                    ax[n,m].set_ylabel(f'{label} NA',labelpad=0.5)

            ax[n,0].text(0.95, 0.95, '-1um', color='white', ha='right', va='top', transform=ax[n,0].transAxes)
            ax[n,1].text(0.95, 0.95, '0um', color='white', ha='right', va='top', transform=ax[n,1].transAxes)
            ax[n,2].text(0.95, 0.95, '+1um', color='white', ha='right', va='top', transform=ax[n,2].transAxes)
            ax[n,3].text(0.95, 0.95, 'XZ', color='white', ha='right', va='top', transform=ax[n,3].transAxes)
            ax[n,4].text(0.95, 0.95, 'YZ', color='white', ha='right', va='top', transform=ax[n,4].transAxes)
        

        plt.tight_layout()
        fig, ax = plt.subplots(2,5,figsize=(6,3))
        stacks = [stack3,stack4]
        pos = [(19,41),(57,60)]

        for n in range(2):
            x0,y0 = pos[n]
            ax[n,0].imshow(stacks[n][0,x0-hw:x0+hw,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,1].imshow(stacks[n][10,x0-hw:x0+hw,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,2].imshow(stacks[n][-1,x0-hw:x0+hw,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,3].imshow(stacks[n][:,x0,y0-hw:y0+hw],cmap='coolwarm')
            ax[n,4].imshow(stacks[n][:,x0-hw:x0+hw,y0],cmap='coolwarm')

            ticks = [0,2]
            for m in range(5):
                ax[n,m].set_xticks(np.linspace(0,19,len(ticks)))
                ax[n,m].set_yticks(np.linspace(0,19,len(ticks)))
                ax[n,m].set_xticklabels(ticks); ax[n,m].set_yticklabels(ticks)
                ax[n,m].set_xlabel('um',labelpad=0.5)
                if m > 0:
                    ax[n,m].set_ylabel('um',labelpad=0.5)
                else:
                    if n == 0:
                        label='High'
                    else:
                        label='Low'
                    ax[n,m].set_ylabel(f'{label} NA',labelpad=0.5)

            ax[n,0].text(0.95, 0.95, '-1um', color='white', ha='right', va='top', transform=ax[n,0].transAxes)
            ax[n,1].text(0.95, 0.95, '0um', color='white', ha='right', va='top', transform=ax[n,1].transAxes)
            ax[n,2].text(0.95, 0.95, '+1um', color='white', ha='right', va='top', transform=ax[n,2].transAxes)
            ax[n,3].text(0.95, 0.95, 'XZ', color='white', ha='right', va='top', transform=ax[n,3].transAxes)
            ax[n,4].text(0.95, 0.95, 'YZ', color='white', ha='right', va='top', transform=ax[n,4].transAxes)
        
        
        plt.tight_layout()
        plt.show()


        

