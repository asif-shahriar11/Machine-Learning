import sys
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import imageio
import os

class GaussianMixture:
    def __init__(self, data, K_range = [3, 4, 5, 6, 7, 8], max_iterations = 5, epsilon = 1e-4, delay = 0.1) -> None:
        self.data = data
        self.K_range = K_range
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.delay = delay

        self.best_log_likelihoods = {} # key: K, value: log likelihood
        self.best_weights = {} # key: K, value: weights
        self.best_means = {} # key: K, value: means
        self.best_covariances = {} # key: K, value: covariances
        self.best_K = -1 # best K

    def init_params(self, K):
        N = self.data.shape[0]
        self.weights = np.random.rand(K)
        self.weights /= np.sum(self.weights)
        self.means = np.random.randn(K, 2) 
        # self.means = self.kmeans_plusplus_initializer(K, self.data) # this should be used - not using here for better visualization
        self.covariances = np.array([np.eye(2) for _ in range(K)])
        self.responsibilities = np.zeros((N, K))

    
    
    def kmeans_plusplus_initializer(self, K, data):
        N = data.shape[0]
        dimensions = data.shape[1]
        centers = np.zeros((K, dimensions))
        
        # Randomly choose the first center
        first_center_idx = np.random.choice(N)
        centers[0] = data[first_center_idx]
        
        for i in range(1, K):
            dist_sq = np.array([min([np.inner(c-x, c-x) for c in centers[:i]]) for x in data])
            probabilities = dist_sq / dist_sq.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers[i] = data[j]
                    break
        
        return centers


    def expectation_step(self):
        """
        returns the responsibilities of each data point for each cluster
        """
        N = self.data.shape[0] # number of data points
        K = self.weights.shape[0] # number of Gaussian components
        for k in range(K):
            self.responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(self.data, mean=self.means[k])#, cov=self.covariances[k])
        self.responsibilities /= np.sum(self.responsibilities, axis=1).reshape(N, 1)


    def maximization_step(self):
        N, D = self.data.shape # number of data points, dimension of data points
        K = self.responsibilities.shape[1] # number of Gaussian components
        Nk = np.sum(self.responsibilities, axis=0) # number of data points assigned to each Gaussian component
        self.weights = Nk / N # proportion of data points assigned to each Gaussian component
        for k in range(K):
            # calculating the mean of each Gaussian component using the data points assigned to it
            self.means[k] = np.sum(self.responsibilities[:, k, np.newaxis] * self.data, axis=0) / Nk[k]
            data_centered = self.data - self.means[k]
            # calculating the covariance of each Gaussian component using the data points assigned to it
            cov_k = (data_centered.T @ (self.responsibilities[:, k, np.newaxis] * data_centered)) / Nk[k]
            self.covariances[k] = cov_k + self.epsilon * np.eye(D)  # Regularizing the covariance of each Gaussian component to avoid singularity issues
        

    def compute_log_likelihood(self):
        """
        compute the log-likelihood of the data points, ie how well the model fits the data, given the Gaussian Mixture Model
        """
        K = self.weights.shape[0] # number of Gaussian components
        likelihood_contributions = np.zeros((self.data.shape[0], K)) # stores the contribution of each data point x_i to the likelihood
        for k in range(K):
            # mathematical formula: pi_k * N(x_i | mu_k, sigma_k)
            likelihood_contributions[:, k] = self.weights[k] * multivariate_normal.pdf(self.data, mean=self.means[k], cov=self.covariances[k], allow_singular=True)

        # mathematical formula: sum_i (log(sum_k (pi_k * N(x_i | mu_k, sigma_k))))
        return np.sum(np.log(np.sum(likelihood_contributions, axis=1)))


    
    def gmm(self, k):
        """
        returns the Gaussian Mixture Model parameters
        """
        self.init_params(k)
        prev_log_likelihood = float('-inf')
        for _ in range(self.max_iterations):
            self.expectation_step()
            self.maximization_step()
            log_likelihood = self.compute_log_likelihood()
            if np.abs(log_likelihood - prev_log_likelihood) < self.epsilon: break
            prev_log_likelihood = log_likelihood
        return self.weights, self.means, self.covariances, log_likelihood
    
    
    def gmm_animation(self, K, num_iterations = 10):
        """
        shows the animation of the GMM estimation at each iteration
        """
        self.init_params(K)

        plt.ion()  # Turn on interactive mode for dynamic updates
        fig, ax = plt.subplots(figsize=(8, 6))

        # creating a folder to save the figures
        if not os.path.exists('gmm'):   os.makedirs('gmm')
            
        for i in range(num_iterations):
            self.expectation_step()
            self.maximization_step()

            log_likelihood = self.compute_log_likelihood()
            #if np.abs(log_likelihood - prev_log_likelihood) < 1e-6: break
                
            ax.clear()  # Clear the plot to update it
            self.plot_gmm_3(ax, K)  # Pass the Axes object to the plotting function
                
            # Update plot title to show current iteration
            plt.title(f'GMM Estimation with K={K}, Iteration: {i+1}')
            plt.draw()
            plt.pause(self.delay)  # Pause to update the plot visually
            # save the figure in the 'gmm' folder
            plt.savefig(f'gmm/gmm_{i+1}.png')

            
        plt.ioff()  # Turn off interactive mode

        # Creating a GIF animation from the saved figures
        images = []
        for i in range(num_iterations): images.append(imageio.imread(f'gmm/gmm_{i+1}.png'))
        imageio.mimsave(f'figures/gmm_{K}.gif', images, fps=2)

        # Deleting the saved figures from the 'gmm' folder
        for i in range(num_iterations): os.remove(f'gmm/gmm_{i+1}.png')
        # Delete the 'gmm' folder
        os.rmdir('gmm')

        return self.weights, self.means, self.covariances, log_likelihood
    
    
    
    def estimate_gmm(self):
        
        best_aic = np.inf # Akaike Information Criterion: penalizes the model complexity to avoid overfitting (smaller is better)
        best_bic = np.inf # Bayesian Information Criterion: similar to AIC but has a larger penalty for bigger models (smaller is better)
        for i in range(len(self.K_range)):
            k = self.K_range[i]          
            self.best_weights[k], self.best_means[k], self.best_covariances[k], self.best_log_likelihoods[k] = self.gmm_animation(k, 20)
            self.plot_gmm_2(k)

            aic, bic = self.compute_aic_bic(self.best_log_likelihoods[k], k, self.data.shape[0], self.data.shape[1])
            if aic < best_aic:
                best_aic = aic
                self.best_K_aic = k
                self.best_k_idx = i
            if bic < best_bic: 
                best_bic = bic
                self.best_K_bic = k
                self.best_k_idx = i

        self.best_K = self.best_K_aic

        print(self.best_K)
        
        print(self.best_log_likelihoods)
        self.plot_likelihoods()
        self.plot_gmm_2(self.best_K)


    def compute_aic_bic(self, log_likelihood, K, N, D):
        """
        computes the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC)
        smaller is better for both AIC and BIC
        """
        aic = -2 * log_likelihood + 2 * (K * (D + 1) + K - 1)
        bic = -2 * log_likelihood + np.log(N) * (K * (D + 1) + K - 1)
        return aic, bic


    def plot_likelihoods(self):
        plt.figure(figsize=(8, 5))
        log_likelihoods = list(self.best_log_likelihoods.values())
        K_range = list(self.best_log_likelihoods.keys())  # Ensure K_range matches the keys from best_log_likelihoods
        
        # Enhanced Plot
        plt.plot(K_range, log_likelihoods, marker='o', markersize=8, linestyle='-', linewidth=2, color='royalblue', label='Log-Likelihood')
        
        # Adding a fill under the curve for better visual impact
        plt.fill_between(K_range, log_likelihoods, alpha=0.1, color='blue')
        
        # Highlighting the best K value
        plt.scatter(K_range[self.best_k_idx], log_likelihoods[self.best_k_idx], color='red', s=100, zorder=5, label='Best K (based on aic)')
        
        # Styling the plot
        plt.title('Log-Likelihood vs. K in GMM (Regularized)', fontsize=12)
        plt.xlabel('K (Number of Gaussian Components)', fontsize=10)
        plt.ylabel('Log-likelihood', fontsize=14)
        plt.xticks(K_range)  # Ensure all K values are shown
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()  # Adjust layout
        
        plt.savefig('figures/log_likelihood_vs_K.png')
        plt.show()



    def plot_gmm_2(self, k):
        """
        for plotting the best GMM without animation
        """

        #k = self.best_K
        weights = self.best_weights[k]
        means = self.best_means[k]
        covariances = self.best_covariances[k]
        plt.figure(figsize=(8, 6))
        # Data points with adjusted edge color for clarity
        plt.scatter(self.data[:, 0], self.data[:, 1], s=40, alpha=0.8, label='Data Points', edgecolor='w', linewidth=0.5, cmap='winter')

        def draw_ellipse(position, covariance, ax=None, **kwargs):
            ax = ax or plt.gca()
            if covariance.shape == (2, 2):
                U, s, Vt = np.linalg.svd(covariance)
                angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                width, height = 2 * np.sqrt(s)
            else:
                angle = 0
                width, height = 2 * np.sqrt(covariance)
            
            ellipse = Ellipse(position, width=width, height=height, angle=angle, **kwargs)
            ax.add_patch(ellipse)
            # Annotate the component number at the mean position
            ax.annotate(f'{i+1}', position, color=ellipse.get_edgecolor(), weight='bold', 
                        horizontalalignment='center', verticalalignment='center')

        colors = plt.cm.viridis(np.linspace(0, 1, k))
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            # Adjusted alpha value for ellipses based on component weights, enhancing visual opacity differentiation
            draw_ellipse(mean, cov, alpha=0.55, color=colors[i], edgecolor='yellow', linewidth=2, label=f'Component {i+1}')

        plt.title(f'Estimated GMM with K={k}', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=10)
        plt.ylabel('Principal Component 2', fontsize=10)
        #plt.legend(loc='upper right', title='Elements')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(f'figures/estimated_gmm_{k}.png')
        plt.show()
            


    def plot_gmm_3(self, ax, k):
        """
        for plotting gmm with animation
        """

        weights = self.weights
        means = self.means
        covariances = self.covariances
            
        # Data points with adjusted edge color for clarity
        ax.scatter(self.data[:, 0], self.data[:, 1], s=30, alpha=0.5, label='Data Points', edgecolor='k', zorder=1)

        def draw_ellipse_2(position, covariance, ax, alpha, color, edgecolor, label=None):
            if covariance.shape == (2, 2):
                U, s, Vt = np.linalg.svd(covariance)
                angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                width, height = 2 * np.sqrt(2.0 * s)
            else:
                angle = 0
                width, height = 2 * np.sqrt(2.0 * covariance)
            
            # Ellipse patch
            ellipse = Ellipse(position, width=width, height=height, angle=angle, facecolor=color, edgecolor=edgecolor, alpha=alpha, label=label)
            ax.add_patch(ellipse)

        colors = plt.cm.viridis(np.linspace(0, 1, k))
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            # Adjusted alpha value for ellipses based on component weights, enhancing visual opacity differentiation
            draw_ellipse_2(mean, cov, ax, alpha=0.55, color=colors[i], edgecolor='yellow', label=f'Component {i+1}')

        ax.set_title(f'Estimated GMM with K={k}', fontsize=14)
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)





        
def pca(D):
    """
    project the data points along the two most prominent principal axes
    :param D: N x M matrix
    :return: U: N x 2 matrix, S: 2 x 2 matrix, V: 2 x M matrix, D: N x 2 matrix
    :U: left singular vectors
    :S: diagonal matrix of singular values
    :V: right singular vectors
    :D: data points projected along the two most prominent principal axes
    """

    # print(D.shape)
    N = D.shape[0]
    M = D.shape[1]
    if M > 2:
        D_centered = D - np.mean(D, axis=0) # centering the data points
        U, S, V_t = np.linalg.svd(D_centered, full_matrices=False)
        D = D_centered @ V_t.T[:, :2] # projecting the data points along the two most prominent principal axes

    plt.figure(figsize=(8, 6))  # Set figure size
    plt.scatter(D[:, 0], D[:, 1], alpha=0.7, edgecolor='w', s=50, c=np.linspace(0, 1, N), cmap='viridis')  # Add color gradient
    plt.title('PCA Projection of Data', fontsize=12)  # Add a title
    plt.xlabel('Principal Component 1', fontsize=10)  # Label x-axis
    plt.ylabel('Principal Component 2', fontsize=10)  # Label y-axis
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid
    plt.axhline(y=0, color='k', linewidth=0.9)  # Add horizontal line at y=0
    plt.axvline(x=0, color='k', linewidth=0.9)  # Add vertical line at x=0
    plt.colorbar(label='Sample Index')  # Add color bar to indicate sample index
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig('figures/pca.png')  # Save the figure
    plt.show()  # Display the plot
    return D



if __name__ == '__main__':

    np.random.seed(1)

    # load the sample dataset file
    # file_path_1 = '2D_data_points_1.txt'
    # file_path_2 = '2D_data_points_2.txt'
    # file_path_3 = '3D_data_points.txt'
    # file_path_4 = '6D_data_points.txt'
    # taking the file name from command line
    file_path = sys.argv[1]
    df = pd.read_csv(file_path, delimiter=',')
    # convert the data frame to a numpy matrix
    D = df.values
    print("D.shape =", D.shape)
    D = pca(D)

    # Task-2
    K_range = [3, 4, 5, 6, 7, 8]
    max_iterations = 10
    epsilon = 1e-4
    delay = 0.1
    gmm = GaussianMixture(D, K_range, max_iterations, epsilon, delay)
    gmm.estimate_gmm()
    