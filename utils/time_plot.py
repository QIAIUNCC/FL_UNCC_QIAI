import matplotlib.pyplot as plt
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Sample data for all groups
local_kermany_resnet = [50, 51, 52, 53, 54]
local_srinivasan_resnet = [4, 5, 6]
local_oct500_resnet = [20, 21, 22, 23, 24]
centralized_resnet = [81, 82, 83, 84, 85, 86, 87]

local_kermany_vit = [67, 68, 69, 70, 71]
local_srinivasan_vit = [5, 6, 7]
local_oct500_vit = [29, 30, 31, 32, 33]
centralized_vit = [116, 117, 118, 119, 120]

# Creating the box plots
plt.figure(figsize=(8, 6))  # Increase the figure size

# Set the background color to a light grey and the grid style to white lines
plt.rcParams['axes.facecolor'] = '#eeeeee'
plt.grid(color='white', linestyle='-', linewidth=2)

# Create box plots with different colors, adjust the positions for compactness
bp1 = plt.boxplot([centralized_resnet, local_kermany_resnet, local_srinivasan_resnet, local_oct500_resnet],
                  positions=[1, 2, 3, 4], widths=0.45, patch_artist=True)
bp2 = plt.boxplot([centralized_vit, local_kermany_vit, local_srinivasan_vit, local_oct500_vit],
                  positions=[1.5, 2.5, 3.5, 4.5], widths=0.45, patch_artist=True)

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
group_labels = ['Centralized', 'Local Kermany', 'Local Srinivasan', 'Local OCT-500']
plt.xticks([1.25, 2.25, 3.25, 4.25], group_labels)
plt.xlim(0.5, 5)
# Add a custom legend with a title
legend_elements = [mpl.lines.Line2D([0], [0], color=color_resnet, lw=4, label='ResNet18'),
                   mpl.lines.Line2D([0], [0], color=color_vit, lw=4, label='ViT')]
plt.legend(handles=legend_elements, title='Models')

# Display the plot with a grid
plt.grid(True)
plt.tight_layout()  # Adjust spacing between subplots if necessary
plt.show()
