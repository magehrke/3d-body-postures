import matplotlib.pyplot as plt

"""
    Author: MA Gehrke
    Date: 26.07.2022
    
    Reorder and name plots that we made
    during the analysis in order to 
    present them in the paper.
"""

# PLOT 1: Main results 1

im1 = plt.imread(f'../output/all/all_viewpoints/barplots/bodypart/bodypart_bar_all-vps.png')
im4 = plt.imread(f'../output/all/all_viewpoints/boxplots/movement/movement_hist_all_vps.png')
im3 = plt.imread(f'../output/all/all_viewpoints/barplots/dailyaction/dailyaction_bar_all-vps.png')
im2 = plt.imread(f'../output/all/all_viewpoints/barplots/emotion/emotion_bar_all-vps.png')

font_size = 9

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(im1)
ax1.axis('off')
ax1.set_title('Most Salient Body Parts', fontsize=font_size)
ax2.imshow(im2)
ax2.set_title('Recognizing an Emotion', fontsize=font_size)
ax2.axis('off')
ax3.imshow(im3)
ax3.axis('off')
ax3.set_title('Recognizing a Daily Action', fontsize=font_size)
ax4.imshow(im4)
ax4.axis('off')
ax4.set_title('Movement', fontsize=font_size)
plt.tight_layout()
plt.savefig(f'../output/final_main_figure_1.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

# PLOT 2: Main results 2

im1 = plt.imread(f'../output/all/all_viewpoints/boxplots/possibility/possibility_hist_all_vps.png')
im2 = plt.imread(f'../output/all/all_viewpoints/boxplots/realism/realism_hist_all_vps.png')

font_size = 9

fig2, ((ax1, ax2)) = plt.subplots(1, 2)
ax1.imshow(im1)
ax1.axis('off')
ax1.set_title('Possiblity of Body Parts', fontsize=font_size)
ax2.imshow(im2)
ax2.set_title('Realism of Posture', fontsize=font_size)
ax2.axis('off')
plt.tight_layout()
plt.savefig(f'../output/final_main_figure_2.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()


# PLOT 3: Consensus
font_size = 8

im1 = plt.imread(f'../output/all/all_viewpoints/barplots/bodypart/bodypart_consensus_max_category_all-vps.png')
im2 = plt.imread(f'../output/all/all_viewpoints/barplots/emotion/emotion_consensus_max_category_all-vps.png')
im3 = plt.imread(f'../output/all/all_viewpoints/barplots/dailyaction/dailyaction_consensus_max_category_all-vps.png')


fig3, ((ax1, ax2, ax3)) = plt.subplots(3, 1)
ax1.imshow(im1)
ax1.axis('off')
ax1.set_title('Most Salient Body Parts', fontsize=font_size)
ax2.imshow(im2)
ax2.set_title('Recognizing an Emotion', fontsize=font_size)
ax2.axis('off')
ax3.imshow(im3)
ax3.axis('off')
ax3.set_title('Recognizing a Daily Action', fontsize=font_size)
plt.tight_layout()
plt.savefig(f'../output/final_consensus_figure.png', bbox_inches='tight', dpi=400)
plt.show()
