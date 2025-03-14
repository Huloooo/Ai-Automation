import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def display_training_info():
    # Create output directory if it doesn't exist
    os.makedirs('analysis_output', exist_ok=True)
    
    # Load and analyze label distribution
    labels_img = mpimg.imread('runs/detect/train4/labels.jpg')
    plt.figure(figsize=(12, 8))
    plt.imshow(labels_img)
    plt.axis('off')
    plt.title('Label Distribution in Dataset')
    plt.savefig('analysis_output/label_distribution.png')
    plt.close()
    
    # Load and analyze label correlogram
    correlogram_img = mpimg.imread('runs/detect/train4/labels_correlogram.jpg')
    plt.figure(figsize=(12, 8))
    plt.imshow(correlogram_img)
    plt.axis('off')
    plt.title('Label Correlations')
    plt.savefig('analysis_output/label_correlations.png')
    plt.close()
    
    # Load and analyze training batch examples
    for i in range(3):
        batch_img = mpimg.imread(f'runs/detect/train4/train_batch{i}.jpg')
        plt.figure(figsize=(12, 8))
        plt.imshow(batch_img)
        plt.axis('off')
        plt.title(f'Training Batch {i+1} Example')
        plt.savefig(f'analysis_output/training_batch_{i+1}.png')
        plt.close()
    
    print("Analysis complete! Check the 'analysis_output' directory for visualizations.")

if __name__ == '__main__':
    display_training_info() 