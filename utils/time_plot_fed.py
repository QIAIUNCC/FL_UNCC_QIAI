import matplotlib.pyplot as plt
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Sample data for all groups
FedAvg_resnet = [67, 65, 64, 67]
FedProx_resnet = [68, 66, 65, 69, 66]
FedSR_resnet = [69, 70, 72, 68]
FedMRI_resnet = [72, 71, 71, 69, 70, 68, 72]
APFL_resnet = [62, 63, 61]

FedAvg_vit = [110, 111, 112, 109, 108, 107]
FedProx_vit = [109, 110, 111, 108, 109]
FedSr_vit = [125, 126, 127, 124, 123, 122]
FedMRI_vit = [112, 112, 113, 114, 110, 109]
APFL_vit = [124, 125, 123, 122, 126]

# Creating the box plots
plt.figure(figsize=(10, 6))  # Increase the figure size

# Set the background color to a light grey and the grid style to white lines
plt.rcParams['axes.facecolor'] = '#eeeeee'
plt.grid(color='white', linestyle='-', linewidth=2)

# Create box plots with different colors, adjust the positions for compactness
bp1 = plt.boxplot([FedAvg_resnet, FedProx_resnet, FedSR_resnet, FedMRI_resnet, APFL_resnet],
                  positions=[1, 2, 3, 4, 5], widths=0.45, patch_artist=True)
bp2 = plt.boxplot([FedAvg_vit, FedProx_vit, FedSr_vit, FedMRI_vit, APFL_vit],
                  positions=[1.5, 2.5, 3.5, 4.5, 5.5], widths=0.45, patch_artist=True)

# Set colors
color_resnet = '#7570b3'  # Purple for ResNet
color_vit = '#1b9e77'  # Green for ViT

for bplot in (bp1, bp2):
    for patch in bplot['boxes']:
        if bplot == bp1:
            patch.set_facecolor(color_resnet)
        elif bplot == bp2:
            patch.set_facecolor(color_vit)

# Adding labels and title
plt.ylabel('Time (seconds)')
plt.title('Time Taken to Train a Single Epoch')

# Customize x-axis tick labels
group_labels = ['FedAvg', 'FedProx', 'FedSR', 'FedMRI', 'APFL']
plt.xticks([1.25, 2.25, 3.25, 4.25, 5.25], group_labels)
plt.xlim(0.5, 6)

# Add a custom legend with a title
legend_elements = [mpl.lines.Line2D([0], [0], color=color_resnet, lw=4, label='ResNet18'),
                   mpl.lines.Line2D([0], [0], color=color_vit, lw=4, label='ViT')]
plt.legend(handles=legend_elements, title='Models')

# Display the plot with a grid
plt.grid(True)
plt.tight_layout()  # Adjust spacing between subplots if necessary
plt.show()
