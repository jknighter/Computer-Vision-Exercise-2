#!/usr/bin/env python
# coding: utf-8

# # Graph Cut Segmentation
# 
# In this part you will implement foreground-background segmentation using *Markov random fields* (MRF).
# 
# ## Recap from the lecture
# A Markov random field is a graphical model that expresses the structure of (input and output) variables. In our case that means we do not only use a color model for foreground and background but also take into account the neighborhood relations of the pixels. This encodes the intuition that neighboring pixels usually belong to the same region.
# 
# The color (or more generally, appearance) models and the neighborhood relations are combined in a so-called *energy function* (or cost function), which is then minimized to obtain an optimal label-assignment.
# 
# Given a structured input (here: image pixel colors) $\mathcal{Y} = \{Y_j|j \in I\}$ we want to find the output (here: labeling) $\mathcal{X} = \{X_j | j \in I\}$ such that
# 
# $$
# \hat{\mathcal{X}} = \arg \min_{\mathcal{X}} E(\mathcal{X}, \mathcal{Y})
# $$
# 
# $$
# E(\mathcal{X}, \mathcal{Y}) = \sum_{j\in I}{\psi_u (X_j,Y_j)} + \sum_{i, j\in I}{\psi_p (X_j,X_j,Y_i,Y_j, i, j)}.
# $$
# 
# The set $I$ contains all possible pixel indices. In our two-label (binary) segmentation case, the label variables must be either 0 (background) or 1 (foreground) $X_j \in \{0, 1\}$.
# 
# The so-called *unary potential* $\psi_u (X_j,Y_j)$ is the cost of assigning the label $X_j$ to a pixel with color $Y_j$. In probabilistic terms, the unary potential is
# 
# $$
# \psi_u (X_j,Y_j)=-\omega_u \cdot \log p(X_j|Y_j),
# $$
# 
# with an appropriate model $p$ for the foreground and the background and a weighting factor $\omega_u$. The unaries encourage labeling each pixel with the label (foreground/background) whose color model is a better fit for that particular pixel.
# 
# The *pairwise potential* $\psi_p$ incorporates the dependencies between pixels. To speed up the computation, the pairwise model is usually restricted to neighboring pixels and is therefore set to zero if the $i$th and $j$th pixels are not direct neighbors in a 4-neighborhood. In our case it written as:
# 
# $$
# \psi_p (X_i,X_j,Y_i,Y_j, i, j)=\omega_p\cdot
# \begin{cases}
# 1,&\text{if }   X_i\neq X_j \text{ and } i,j \text{ are neighbors}  \\
# 0,&\text{otherwise}
# \end{cases}
# $$
# 
# with some weighting factor $\omega_p$. Such pairwise potentials encourage neighboring pixels to have the same label.
# 
# A Graph Cut method is used to find the optimal solution $\hat{\chi}$ of the energy function.
# 
# ## Bird's eye overview
# 
# It's easy to get lost in all the details, so here's an roadmap of what we're going to do:
# 
# 1. Set up the Markov Random Field (define unaries and pairwise potentials), in more detail:
#   1. Model the distribution of background and foreground colors based on the colors found in approximate initial regions
#   2. For each pixel independently, calculate the posterior probability of being foreground, based on the models from the previous step ("probability map")
#   3. Calculate the unary potentials based on the foreground probability map
#   4. Define the pairwise potentials (using the neighborhood relations)
# 2. Use graph cuts to minimize the energy function of the Markov Random Field and obtain a labeling
# 
# You will not have to implement the min-cut algorithm yourself, we will use the `pygco` ("Python Graph Cut Optimizer") package for that. You can install it using `conda install pygco -c kayarre`. (This is a Python wrapper over the C++ gco library, which can be found at https://vision.cs.uwaterloo.ca/code/)

# In[ ]:


# get_ipython().run_cell_magic('html', '', '<!-- Add heading numbers -->\n<style>\nbody {counter-reset: section;}\nh2:before {counter-increment: section;\n           content: counter(section) " ";}\n</style>')
#
#
# # In[ ]:
#
#
# # Some imports and helper functions
# get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import cv2
import pygco  # conda install pygco -c kayarre

def draw_mask_on_image(image, mask, color=(0, 255, 255)):
    """Return a visualization of a mask overlaid on an image."""
    result = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, kernel)
    outline = dilated > mask
    result[mask == 1] = (result[mask == 1] * 0.4 + 
                         np.array(color) * 0.6).astype(np.uint8)
    result[outline] = color
    return result


# ## Initial masks
# 
# First, manually create initial boxes of foreground and background regions.
# 
# We will use these to build color models. That is, to model the probability of a pixel color occuring, given either that it is a foreground or a background pixel.

# In[ ]:


im = imageio.imread('../lotus320.jpg')
h,w = im.shape[:2]

# Set up initial foreground and background
# regions for building the color model
init_fg_mask = np.zeros([h, w])
init_bg_mask = np.zeros([h, w])

# Do something like the following but with sensible
# indices
# init_fg_mask[10:20, 15:30] = 1
# init_bg_mask[60:90, 50:110] = 1

# YOUR CODE HERE
init_fg_mask[100:220, 140:260] = 1
init_bg_mask[0:80, 0:80] = 1

# x, y, z = (im[100,200,:]/2.5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].set_title('Initial foreground mask')
axes[0].imshow(draw_mask_on_image(im, init_fg_mask))
axes[1].set_title('Initial background mask')
axes[1].imshow(draw_mask_on_image(im, init_bg_mask))
fig.tight_layout()
plt.show()

# ## Color histograms
# 
# Usually *Gaussian mixture models* are used to model color distributions. However, to keep this exercise simple, we will only use color histograms (i.e. the relative frequencies of quantized colors) in the respective region of the image defined by the boxes. In other words, we model the color simply as a categorical random variable.
# 
# Implement the function `calculate_histogram`. It should take as input the image `img` with values in $[0, 255]$ and a `mask` the same size as the image. The mask is 1 at the positions of the image where the histogram should be computed, zero otherwise. The final parameter `n_bins` defines how many bins should be used in the histogram in each dimension. The function should **return a 3-dimensional array** of shape `[n_bins, n_bins, n_bins]`, containing the relative frequency for each (r,g,b) color bin within the region of the image defined by the mask, i.e. the fraction of pixels falling within each bin. The histogram should be normalized (sum to 1). Initialize all bins with a small value ($10^{−3}$) to counteract relative frequencies which are zero (this method is called additive smoothing).

# In[ ]:


def calculate_histogram(img, mask, n_bins):
    histogram = np.full((n_bins, n_bins, n_bins), fill_value=0.001)

    # YOUR CODE HERE
    # get interested region
    region = img[mask.astype(bool)]
    # count frequencies
    reg_h, reg_w = region.shape[:2]
    for color in region:
        idx_r, idx_g, idx_b = ((color / 255) * n_bins).astype(int) - 1
        histogram[idx_r, idx_g, idx_b] += 1
    # normalization
    histogram += histogram / np.sum(histogram)
    return histogram


# In[ ]:


n_bins = 10
fg_histogram = calculate_histogram(im, init_fg_mask, n_bins)
bg_histogram = calculate_histogram(im, init_bg_mask, n_bins)

fig, axes = plt.subplots(
    3, 2, figsize=(5,5), sharex=True, 
    sharey=True, num='Relative frequency of color bins')

x = np.arange(n_bins)
axes[0,0].bar(x, np.sum(fg_histogram, (1, 2)))
axes[0,0].set_title('red (foreground)')
axes[1,0].bar(x, np.sum(fg_histogram, (0, 2)))
axes[1,0].set_title('green (foreground)')
axes[2,0].bar(x, np.sum(fg_histogram, (0, 1)))
axes[2,0].set_title('blue (foreground)')

axes[0,1].bar(x, np.sum(bg_histogram, (1, 2)))
axes[0,1].set_title('red (background)')
axes[1,1].bar(x, np.sum(bg_histogram, (0, 2)))
axes[1,1].set_title('green (background)')
axes[2,1].bar(x, np.sum(bg_histogram, (0, 1)))
axes[2,1].set_title('blue (background)')
fig.tight_layout()
plt.show()

# What do you see in these histograms?

# YOUR ANSWER HERE

# ## Foreground probability map
# 
# The next step in the segmentation process is to estimate a probability map: For each pixel we want to estimate the probability that it belongs to the foreground. This will be used as basis for the unary potential.
# 
# The function `foreground_pmap(img, fg_histogram, bg_histogram)` should take the image `img` and the two histograms `fg_histogram`, `bg_histogram` estimated from the foreground region and the background region respecively. It should return an array of shape $\texttt{height}\times\texttt{width}$ containing the probability of each pixel belonging to the foreground. To estimate the required probability $p(c|[r, g, b])$ from the computed histograms, a class prior $p(c)$ of $0.5$ should be used, which means that both foreground and background pixels are equally likely a priori. 
# 
# Recall Bayes' theorem applied to this case:
# 
# $$
# p(c\ |\ r,g,b) = \frac{p(c) \cdot p(r,g,b\ |\ c)}{p(r,g,b)} = \frac{p(c)\cdot p(r,g,b\ |\ c)}{\sum_{\tilde{c}} p(\tilde{c}) \cdot p(r,g,b\ |\ \tilde{c}) }
# $$

# In[ ]:


def foreground_pmap(img, fg_histogram, bg_histogram):
    # YOUR CODE HERE
    priori_fg = 0.5
    priori_bg = 1 - priori_fg
    pmap = np.zeros(img.shape[:2])
    im_h, im_w = img.shape[:2]
    for i in range(im_h):
        for j in range(im_w):
            idx_r, idx_g, idx_b = ((img[i, j] / 255) * n_bins).astype(int) - 1
            likelihood_fg = fg_histogram[idx_r][idx_g][idx_b]
            likelihood_bg = bg_histogram[idx_r][idx_g][idx_b]
            pmap[i, j] = (priori_fg * likelihood_fg) / (priori_fg * likelihood_fg + priori_bg * likelihood_bg)
    return pmap

# In[ ]:


foreground_prob = foreground_pmap(im, fg_histogram, bg_histogram)
fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=True)
axes[0].imshow(im)
axes[0].set_title('Input image')
im_plot = axes[1].imshow(foreground_prob, cmap='viridis')
axes[1].set_title('Foreground posterior probability')
fig.tight_layout()
fig.colorbar(im_plot, ax=axes)
foreground_map = (foreground_prob > 0.5)
plt.show()


# Explain what you see in the probability map.

# YOUR ANSWER HERE

# ## Unary potentials
# Use the previously computed probability map `foreground_map` to compute the unary potential for both foreground and background.
# 
# This function `unary_potentials(probability_map, unary_weight)` shall use the `probability_map` and a scalar weighting factor to compute the unary potentials. It should return a matrix of the same size as the probability matrix.

# In[ ]:


def unary_potentials(probability_map, unary_weight):
    # YOUR CODE HERE
    return - np.log(probability_map) * unary_weight
    
unary_weight = 1
unary_fg = unary_potentials(foreground_prob, unary_weight)
unary_bg = unary_potentials(1 - foreground_prob, unary_weight)
fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=True)
axes[0].imshow(unary_fg)
axes[0].set_title('Unary potentials (foreground)')
im_plot = axes[1].imshow(unary_bg)
axes[1].set_title('Unary potentials (background)')
fig.tight_layout()
fig.colorbar(im_plot, ax=axes)
plt.show()

# Why are the unary potentials for the foreground so small in the middle of the flower?

# YOUR ANSWER HERE

# ## Pairwise potentials
# 
# Create a function to compute the prefactor $w_p$ of the pairwise potential for two specific pixels. Implement the funtion below, where `img` is the image, `(x1, y1), (x2, y2)` are the pixel coordinates in the image and the last parameter is the weight $\omega_p$ for the pairwise potential. (Do not confuse `(x1, y1), (x2, y2)` with the $X_j, Y_j$ from the top of the page) 
# 
# Keep in mind that this prefactor does not depend on the labels and is therefore independent of $\mathcal{X}$.
# 
# Also, the function signature is more general (see the contrast-sensitive Potts Model question later on), not all parameters are needed here.
# 
# *Hint:* the function is extremely simple.

# In[ ]:


def pairwise_potential_prefactor(img, x1, y1, x2, y2, pairwise_weight):
    # YOUR CODE HERE
    if x1 == x2 and y1 == y2:
        potential = 0
    else:
        potential = pairwise_weight
    return potential


# Using the functions from the previous task, implement a function to compute all the pairwise potentials for the image using 4-neighborhoods. That means only the top, bottom, left and right neighboring pixels should be connected to a given pixel.
# 
# The function `pairwise_potentials` should return the `edges` (represented as index pairs) and an array `costs` containing the corresponding edge costs (i.e. the value of the pairwise potential prefactor). Note that you have to use a linearized index instead of (x,y)-coordinates. A conversion function is supplied (`coords_to_index(x, y, width)`).
# 
# Again, `edges` should be an integer array of shape $k\times 2$, while `costs` should have length $k$, where $k$ is the number of neighborhood-edges in the image grid.

# In[ ]:


def coords_to_index(x, y, width):
    return y * width + x

def pairwise_potentials(im, pairwise_weight):
    # YOUR CODE HERE
    im_h, im_w = im.shape[:2]
    for y in range(im_h):
        for x in range(im_w):

    return edges, costs

pairwise_edges, pairwise_costs = pairwise_potentials(im, pairwise_weight=3)


# Now you can execute the optimization and plot the resulting labeling.

# In[ ]:


def graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs):
    unaries = np.stack([unary_bg.flat, unary_fg.flat], axis=-1)
    labels = pygco.cut_gencut_general_graph(
        pairwise_edges, pairwise_costs, unaries, 
        1-np.eye(2), n_iter=-1, algorithm='swap')
    return labels.reshape(unary_fg.shape)

graph_cut_result = graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs)
fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=True)
axes[0].set_title('Simple thresholding of foreground probability')
axes[0].imshow(draw_mask_on_image(im, foreground_prob>0.5))
axes[1].set_title('Graph cut result')
axes[1].imshow(draw_mask_on_image(im, graph_cut_result))
fig.tight_layout()


# Why is the segmentation the way it is?

# YOUR ANSWER HERE

# ## [BONUS] Try another image
# 
# First, create a single function that runs the whole segmentation pipeline starting from the image.
# 
# `segment_image(...)` should return the final binary segmentation mask with 1 at the foreground pixels and 0 at the background.

# In[ ]:


def segment_image(im, init_fg_mask, init_bg_mask,
                  unary_weight, pairwise_weight, n_bins):
    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:


import skimage.data

def run_on_another_image():
    im = skimage.data.immunohistochemistry()
    #im = imageio.imread('flowers.jpg')
    #im = skimage.data.stereo_motorcycle()[0]
    h, w = im.shape[:2]
    fg_mask = np.zeros([h, w])
    bg_mask = np.zeros([h, w])

    # Set some appropriate parts of fg_mask and bg_mask to 1 for initialization.
    # YOUR CODE HERE
    raise NotImplementedError()

    graph_cut_result = segment_image(
        im, fg_mask, bg_mask, 
        unary_weight=1, pairwise_weight=1, n_bins=8)
    
    fig, axes = plt.subplots(1, 3, figsize=(14,5), sharey=True)
    axes[0].set_title('Initial foreground mask')
    axes[0].imshow(draw_mask_on_image(im, fg_mask))
    axes[1].set_title('Initial background mask')
    axes[1].imshow(draw_mask_on_image(im, bg_mask))
    axes[2].set_title('Graph cut result')
    axes[2].imshow(draw_mask_on_image(im, graph_cut_result))
    fig.tight_layout()
    
run_on_another_image()


# Does it look good? Which parameter would you need to change to reduce the number of holes in the segmentation? Try it.
# 
# Now try segmenting `im = skimage.data.stereo_motorcycle()[0]` using this technique. Can you segment out the motorbike by fiddling with the parameters? Why or why not?

# ## [BONUS] Contrast-Sensitive Potts Model
# Go back to the `pairwise_potential_prefactor` function and modify it to incorporate a new term, resulting in the so-called *contrast sensitive Potts model*. The new pairwise potential should be:
# 
# $$
# \psi_p (X_i,X_j,Y_i,Y_j,i,j)=\omega_p\cdot \exp\left(-\omega_d\|Y_i - Y_j\|^2\right)\cdot 
# \begin{cases}
# 1,&\text{if } X_i \neq X_j \text{ and } i,j \text{ are neighbors}  \\
# 0,&\text{otherwise}
# \end{cases}
# $$
# 
# This means the prefactor is now $\omega_p\cdot \exp\left(-\omega_d\|Y_i - Y_j\|^2\right)$. For simplicity, you can hardcode the parameter $\omega_d$. 

# What changes when using the contrast sensitive Potts model? What is the intuition behind adding this new term?

# YOUR ANSWER HERE

# ## [BONUS] Iterative Segmentation
# 
# We can make the result better if we iterate the labeling process several times. Implement `iterative_opt`, a method to execute the optimization process iteratively. 
# 
# Use the previously computed labeling as initial segmentation (instead of the rectangular masks we used above) and estimate new models (histograms and unaries) for foreground and background based on these. Solve the graph cut problem and use the resulting segmentation in the next iteration.

# In[ ]:


def iterative_opt(img, fg_mask, n_bins, unary_weight,
                  pairwise_edges, pairwise_costs, n_iter):
    # YOUR CODE HERE
    raise NotImplementedError()

labels_5 = iterative_opt(
    im, graph_cut_result, n_bins, unary_weight, pairwise_edges, pairwise_costs, n_iter=5)
labels_10 = iterative_opt(
    im, labels_5, n_bins, unary_weight, pairwise_edges, pairwise_costs, n_iter=5)

fig, axes = plt.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True)
axes[0].set_title('Initial')
axes[0].imshow(draw_mask_on_image(im, graph_cut_result))
axes[1].set_title(f'After 5 iterations')
axes[1].imshow(draw_mask_on_image(im, labels_5))
axes[2].set_title(f'After 10 iterations')
axes[2].imshow(draw_mask_on_image(im, labels_10))
fig.tight_layout()


# How did the labeling change? Do you have an explanation why?

# YOUR ANSWER HERE

# ## [BONUS++] Interactive Segmentation
# 
# We can get even better results by incorporating user input into the iterative segmentation process you implemented above.
# 
# Extend the given framework to allow the user to add or remove rectangular regions from the graph cut result. Then recalculate the foreground and background model according to new mask. Iterate this process until the user is satisfied with the result.
# 
# For this, look up how to create interactive graphical interfaces using Matplotlib, see for example `matplotlib.widgets.RectangleSelector` and `matplotlib.widgets.Button`.

# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()

