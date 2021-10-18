from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from stl10_input import get_dataset

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


SIFT = cv2.SIFT_create()
CLASS_MAP = {
    1: "airplane",
    2: "bird",
    9: "car",
    7: "horse",
    3: "ship"
}
CLASS_NAMES = list(CLASS_MAP.values())


def detect_kps(img: np.ndarray) -> list:
    return SIFT.detect(img)


def detect_kps_descs(img: np.ndarray) -> tuple:
    return SIFT.detectAndCompute(img, None)


def sample_feature_extraction(X: np.ndarray, Y: np.ndarray, show: bool = False):
    os.makedirs("./img/", exist_ok=True)
    for c_ind, label in enumerate(np.unique(Y)):
        curr_class = np.where(Y == label)[0]

        _, axs = plt.subplots(1, 2, figsize=(10,5))
        for i_ind in range(2):
            # get random image from train set
            img = X[np.random.choice(curr_class)]

            # detect and plot keypoints
            kps = detect_kps(img)
            x = [kp.pt[0] for kp in kps]
            y = [kp.pt[1] for kp in kps]
            s = [kp.size for kp in kps]
            s /= np.min(s) * 0.1
            axs[i_ind].scatter(x, y, facecolors='none', edgecolors='r', s=s)
            
            # show image
            axs[i_ind].imshow(img)
            axs[i_ind].set_axis_off()

        plt.suptitle(r"$\bf{Class-}$" + CLASS_NAMES[c_ind], fontsize='xx-large')
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(f"./img/{CLASS_NAMES[c_ind]}")
    
    plt.close('all')


def build_visual_vocabulary(X: np.ndarray, n_clusters: int) -> KMeans:
    print("Building visual vocabulary:")
    print("[1/2] Extracting descriptors")
    descriptors = []
    for img in X:
        _, descs = detect_kps_descs(img)
        if descs is not None:
            descriptors.extend(descs)
    
    print("[2/2] Clustering descriptors")
    return KMeans(n_clusters=n_clusters, random_state=42).fit(descriptors)


def match_clusters(X: np.ndarray, km: KMeans) -> list:
    clusters = []
    for img in X:
        _, descs = detect_kps_descs(img)
        if descs is not None:
            descs = np.array(descs, dtype=float)
            clusters.append(km.predict(descs))
        else:
            clusters.append([])

    return clusters


def visualize_words(centers: np.ndarray):
    print("Visualizing words:")

    print("[1/2] t-SNE")
    c2d_tSNE = TSNE(n_components=2, random_state=42).fit_transform(centers)

    print("[2/2] PCA")
    c2d_PCA = PCA(n_components=2, random_state=42).fit_transform(centers)

    _, axs = plt.subplots(1, 2, figsize=(10,5))
    for i, (c2d, method) in enumerate(zip((c2d_tSNE, c2d_PCA), ('t-SNE', 'PCA'))):
        axs[i].scatter(c2d[:,0], c2d[:,1])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].set_title(method)
        axs[i].grid()

    plt.suptitle("Words 2D visualization", fontsize='x-large')
    plt.tight_layout()
    plt.show()


def freq_representation(X: np.ndarray, Y: np.ndarray, vocab_km: KMeans):
    print("Converting classes to frequency representation")

    Y_unique = np.unique(Y)
    class_count = len(Y_unique)
    cluster_count = len(vocab_km.cluster_centers_)
    cluster_labels = range(cluster_count)

    nrows = int(np.ceil(class_count / 2))
    ncols = 2
    _, axs = plt.subplots(nrows, ncols, figsize=(5*nrows, 5*ncols), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for c_ind, label in enumerate(Y_unique):
        X_curr = X[Y == label]

        class_counts = get_descriptor_counts(X_curr, vocab_km, normalize=False)
        class_counts = np.sum(class_counts, axis=0)
        class_counts /= np.sum(class_counts) / 100
        
        axs[c_ind].yaxis.set_major_formatter(PercentFormatter())
        axs[c_ind].bar(cluster_labels, class_counts)
        axs[c_ind].set_xticks([])
        axs[c_ind].set_title(r"$\bf{Class-}$" + CLASS_NAMES[c_ind])
    
    # hide axes for remaining subplots
    for i in range(class_count, len(axs)):
        axs[i].set_axis_off()
    
    # TODO: add labels?
    plt.tight_layout()
    plt.show()


def get_descriptor_counts(X: np.ndarray, vocab_km: KMeans, normalize: bool = True) -> list:
    desc_counts = []

    # accumulate counts for each cluster/bin
    for img_clusters in match_clusters(X, vocab_km):
        cluster_counts = np.array([0.] * len(vocab_km.cluster_centers_))
        labels, counts = np.unique(img_clusters, return_counts=True)
        for i, lbl in enumerate(labels):
            cluster_counts[lbl] += counts[i]

        # normalize
        if normalize and np.sum(cluster_counts) != 0:
            cluster_counts /= np.sum(cluster_counts) / 100
        
        desc_counts.append(cluster_counts)
    
    return desc_counts


def build_svm(X: np.ndarray, Y: np.ndarray, vocab_km: KMeans) -> SVC:
    print("Training SVM classifier")
    X_train = get_descriptor_counts(X, vocab_km)
    return SVC(random_state=42).fit(X_train, Y)


def evaluate_svm(svm: SVC, x: np.ndarray, y: np.ndarray):
    print("Evaluating classifier")
    x_desc = get_descriptor_counts(x, vocab_km)

    y_pred = svm.predict(x_desc)
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    avg_precisions = []
    y_pred_conf = svm.decision_function(x_desc)
    for class_ind, class_label in enumerate(np.unique(y)):
        print(f"Evaluating class {class_label} ({CLASS_NAMES[class_ind]})")
        class_conf = y_pred_conf[:, class_ind]
        pred_sort = np.argsort(class_conf)[::-1]

        avg_precision = 0
        correct_pred = 0
        for i, pred_ind in enumerate(pred_sort):
            true_class = y[pred_ind]
            prediction = y_pred[pred_ind]
            if true_class == prediction and prediction == class_label:
                correct_pred += 1
                avg_precision += correct_pred / (i + 1)
        
        total_in_class = len(y[y == class_label])
        avg_precision /= total_in_class
        avg_precisions.append(avg_precision)
        print(f"Average precision: %.2f" % avg_precision)
        print()

        top5_ind = pred_sort[:5]
        bot5_ind = pred_sort[-5::-1]

        _, axs = plt.subplots(2, 5, figsize=(25, 10))
        plt.suptitle(r"$\bf{Classifier-}$" + CLASS_NAMES[class_ind], fontsize='x-large')
        for i, ax in enumerate(axs[0]):
            if i == 2:
                ax.set_title("Top 5 results")
            
            ax.set_axis_off()
            ax.imshow(x[top5_ind[i]])
        
        for i, ax in enumerate(axs[1]):
            if i == 2:
                ax.set_title("Worst 5 results")
            
            ax.set_axis_off()
            ax.imshow(x[bot5_ind[i]])
        
        plt.tight_layout()
        plt.show()
    
    print("mAP: %.2f" % np.mean(avg_precisions))


if __name__ == "__main__":
    np.random.seed(42)
    X_train, Y_train = get_dataset(train_set=True)
    x_test, y_test = get_dataset(train_set=False)

    # sample_feature_extraction(X_train, Y_train, show=True)

    subset_size = 0.5

    assert subset_size <= 0.5
    X_build, X_calc, Y_build, Y_calc = \
        train_test_split(X_train, Y_train, train_size=subset_size, random_state=42, stratify=Y_train)

    for n_clusters in (500, 1000, 2000):

        ###
        vocab_km = build_visual_vocabulary(X_build, n_clusters)
        with open(f"km{n_clusters}.pkl", "wb") as f:
            print("Saving KMeans model")
            pickle.dump(vocab_km, f)
        
        ########
        ## OR ##
        ########

        # with open(f"km{n_clusters}.pkl", "rb") as f:
        #     print("Loading KMeans model")
        #     vocab_km = pickle.load(f)
        ###

        visualize_words(vocab_km.cluster_centers_)
        freq_representation(X_calc, Y_calc, vocab_km)

        svm = build_svm(X_calc, Y_calc, vocab_km)
        evaluate_svm(svm, x_test, y_test)
        # evaluate_svm(svm, x_test[:100], y_test[:100])
