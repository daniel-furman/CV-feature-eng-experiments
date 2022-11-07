"""
Helper functions to run plotting visualizations
"""

# general libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def distribution_plots_between_sets(train_images, val_images, test_images):
    # add mean/std variation plots

    img_means_train = []
    img_stds_train = []
    plot_groups_train = []

    for itr, img in enumerate(train_images):
        #if train_labels[itr] == 0: 
        img_means_train.append(np.mean(img))
        img_stds_train.append(np.std(img))
        plot_groups_train.append('train set')

    img_means_val = []
    img_stds_val = []
    plot_groups_val = []

    for itr, img in enumerate(val_images):
    #if val_labels[itr] == 0: 
        img_means_val.append(np.mean(img))
        img_stds_val.append(np.std(img))
        plot_groups_val.append('val set')

    img_means_test = []
    img_stds_test = []
    plot_groups_test = []

    for itr, img in enumerate(test_images):
        img_means_test.append(np.mean(img))
        img_stds_test.append(np.std(img))
        plot_groups_test.append('test set')


    train_sample = np.random.randint(0, len(img_means_train), size=1500)
    val_sample = np.random.randint(0, len(img_means_val), size=1500)
    test_sample = np.random.randint(0, len(img_means_test), size=1500)

    img_means_train = np.array(img_means_train)[train_sample]
    img_stds_train = np.array(img_stds_train)[train_sample]
    plot_groups_train = np.array(plot_groups_train)[train_sample]

    img_means_val = np.array(img_means_val)[val_sample]
    img_stds_val = np.array(img_stds_val)[val_sample]
    plot_groups_val = np.array(plot_groups_val)[val_sample]

    img_means_test = np.array(img_means_test)[test_sample]
    img_stds_test = np.array(img_stds_test)[test_sample]
    plot_groups_test = np.array(plot_groups_test)[test_sample]

    img_means = list(img_means_train) + list(img_means_val) + list(img_means_test)
    img_stds = list(img_stds_train) + list(img_stds_val) + list(img_stds_test)
    plot_groups = list(plot_groups_train) + list(plot_groups_val) + list(plot_groups_test)

    pd_df = pd.DataFrame([img_means, img_stds, plot_groups]).T
    pd_df.columns = ['means', 'stds', 'dataset group\nn=1500 random sample']
    pd_df

    sns.kdeplot(data=pd_df, x="means", hue="dataset group\nn=1500 random sample").set(title='Flattened Image Means Across Splits')

    plt.figure()

    sns.kdeplot(data=pd_df, x="stds", hue="dataset group\nn=1500 random sample").set(title='Flattened Image Standard Deviations Across Splits')


# pca = PCA(2) # we need 2 principal components.
# converted_data = pca.fit_transform(data) 
# converted_data.shape
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize = (10,6))
# c_map = plt.cm.get_cmap('jet', 5)
# plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
#            cmap = c_map , c = train_labels)
# plt.colorbar()
# plt.xlabel('PC-1') , plt.ylabel('PC-2')
# plt.show()