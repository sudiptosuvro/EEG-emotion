# EEG-based Emotion Classification using NN

## Steps for classification:

#### Data Loading and Exploration:
- Imported necessary libraries (NumPy, Pandas, Matplotlib, Seaborn, Plotly, SciPy, TensorFlow).
- Loaded the emotion dataset from a CSV file ('emotions.csv').
- Displayed the first 5 rows of the dataset.
#### Label Conversion:
- Converted textual emotion labels ('NEGATIVE', 'NEUTRAL', 'POSITIVE') to numerical values (0, 1, 2).
#### Emotion Distribution Visualization (Pie Chart):
- Counted the occurrences of each emotion.
- Created a pie chart to visualize the distribution of emotions.
#### Time-Series Visualization:
- Plotted a sample of EEG time-series data.
#### Spectral Analysis (Power Spectral Density):
- Used Welch's method to calculate the power spectral density.
- Plotted the power spectral density.
#### Correlation Heatmap:
- Calculated the correlation matrix.
- Visualized the correlation matrix using a heatmap.
#### t-SNE Visualization:
- Applied t-distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.
- Visualized the data in a 2D scatter plot.
#### Feature Significance Analysis:
- Performed t-tests to identify significant features for each emotion.
- Visualized the number of significant and non-significant features for each emotion using a bar chart.
#### Advanced Preprocessing:
- Normalized the data using z-score normalization.
#### Data Splitting:
- Split the data into training and testing sets.
#### Neural Network Model Building:
- Built a neural network model using TensorFlow and Keras.
- Compiled the model with the Adam optimizer and sparse categorical crossentropy loss.
#### Confusion Matrix:
![confusion matrix](https://github.com/sudiptosuvro/EEG-emotion/assets/147235323/0e302f64-7a01-464e-aa9d-01362ba28752)
