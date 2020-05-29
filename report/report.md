Michał Naruniec\
360386

# Deep Neural Networks
## Assignment 3

### Data analysis

This is a table with 10 samples from each class in the test set.
The arrows progress from the lightests (yellow) to the darkest one (black).

It's hard to notice a strong pattern, but it seems like the classes might represent moving towards particular corner of the lattice.
Class 0 seems to be trending towards the top-right corner, class 1 - top-left, class 2 - bottom-right, class 3 - bottom-left.

| Class 0 | Class 1 | Class 2 | Class 3 |
|:---:|:---:|:---:|:---:|
| ![](img/class_0/1.png) | ![](img/class_1/2.png) | ![](img/class_2/0.png) | ![](img/class_3/5.png) |
| ![](img/class_0/3.png) | ![](img/class_1/4.png) | ![](img/class_2/16.png) | ![](img/class_3/6.png) |
| ![](img/class_0/9.png) | ![](img/class_1/7.png) | ![](img/class_2/20.png) | ![](img/class_3/8.png) |
| ![](img/class_0/10.png) | ![](img/class_1/11.png) | ![](img/class_2/24.png) | ![](img/class_3/12.png) |
| ![](img/class_0/22.png) | ![](img/class_1/13.png) | ![](img/class_2/25.png) | ![](img/class_3/15.png) |
| ![](img/class_0/23.png) | ![](img/class_1/14.png) | ![](img/class_2/32.png) | ![](img/class_3/19.png) |
| ![](img/class_0/27.png) | ![](img/class_1/17.png) | ![](img/class_2/33.png) | ![](img/class_3/21.png) |
| ![](img/class_0/31.png) | ![](img/class_1/18.png) | ![](img/class_2/35.png) | ![](img/class_3/28.png) |
| ![](img/class_0/40.png) | ![](img/class_1/26.png) | ![](img/class_2/37.png) | ![](img/class_3/29.png) |
| ![](img/class_0/41.png) | ![](img/class_1/30.png) | ![](img/class_2/44.png) | ![](img/class_3/34.png) |


### Architecture

The network consists of two stacked layers of LSTM with 8 features in the hidden state, as well as a fully connected layer, transforming 8 features from last LSTM output to 4 class output.
I normalize the classes with softmax and use log-loss for the classification task.
There is additionally an embedding layer at the beginning for the second version of the assignment.

### Evaluation and embedding analysis

The network achieves around 69% accuracy on the test set both with and without embeddings.


#### Embedding analysis

| Original lattice | Trained embeddings|
|:---:|:---:|
| ![](img/final_embedding/ground_truth.png) | ![](img/final_embedding/embed_Snap_a6920_28_05_2020_17_54.png) |

In the visualization of the original lattice above, the corners have red, green, black and yellow colors, and the rest of nodes represent a fluid transition between them.
As you can see, the embeddings almost took a shape of a slightly shifted square. It's rotated and a bit like a rhombus, but it is okay, as the orientation (and probably all non-reductive linear transformations) should not matter for the task.
The important thing is that the color pattern of the points resembles the original one.

Below you can see the history of learning the embeddings as a GIF.

![](img/embedding_history/progress.gif)