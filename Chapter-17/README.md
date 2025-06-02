# Chapter 17: Image Processing Techniques

## Learning Objectives
By the end of this chapter, students will be able to:
- Apply fundamental image enhancement and filtering techniques
- Implement edge detection and feature extraction algorithms
- Perform geometric transformations and morphological operations
- Use histogram analysis for image analysis and enhancement
- Build practical image processing pipelines for real-world applications

## Table of Contents
1. [Introduction to Image Processing](#introduction)
2. [Image Enhancement Techniques](#enhancement)
3. [Spatial Domain Filtering](#spatial-filtering)
4. [Frequency Domain Processing](#frequency-domain)
5. [Edge Detection and Feature Extraction](#edge-detection)
6. [Morphological Operations](#morphological)
7. [Geometric Transformations](#geometric)
8. [Histogram Processing](#histogram)

## 1. Introduction to Image Processing {#introduction}

Image processing involves the manipulation and analysis of digital images to enhance their quality, extract information, or prepare them for further analysis. Unlike computer vision, which focuses on understanding and interpreting images, image processing concentrates on transforming images through mathematical operations.

### Digital Image Processing Pipeline

**Image Acquisition**: Capturing or obtaining digital images from various sources (cameras, scanners, medical devices, satellites).

**Preprocessing**: Initial cleaning and preparation of images including noise reduction, geometric correction, and format standardization.

**Enhancement**: Improving image quality through techniques like contrast adjustment, sharpening, and brightness correction.

**Segmentation**: Partitioning images into meaningful regions or objects for further analysis.

**Feature Extraction**: Identifying and measuring specific characteristics like edges, corners, textures, and shapes.

**Analysis and Interpretation**: Using extracted features for classification, recognition, measurement, or other analytical tasks.

### Applications Across Domains

**Medical Imaging**: Enhancing X-rays, MRIs, and CT scans for better diagnosis, noise reduction in ultrasound images, and automated analysis of medical images.

**Remote Sensing**: Processing satellite and aerial imagery for environmental monitoring, agricultural assessment, urban planning, and disaster response.

**Industrial Quality Control**: Automated inspection of manufactured products, defect detection, surface analysis, and dimensional measurements.

**Digital Photography**: Image enhancement for consumer applications, artistic effects, panorama stitching, and high dynamic range (HDR) imaging.

**Security and Surveillance**: License plate recognition, facial image enhancement, forensic image analysis, and object tracking in video streams.

### Image Processing vs Computer Vision

**Image Processing**: Focuses on transforming images (input: image → output: enhanced/modified image)
**Computer Vision**: Focuses on understanding images (input: image → output: interpretation/decisions)

Many applications combine both approaches in integrated systems.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import data, filters, feature, morphology, exposure, transform, segmentation
from skimage.color import rgb2gray, rgb2hsv
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class ImageProcessingDemo:
    """Comprehensive demonstration of image processing techniques"""
    
    def __init__(self):
        # Load various sample images for demonstration
        self.images = {
            'camera': data.camera(),
            'coins': data.coins(),
            'astronaut': data.astronaut(),
            'checkerboard': data.checkerboard(),
            'text': data.text(),
            'coffee': data.coffee()
        }
        
        # Create noisy versions for enhancement demos
        self.create_noisy_images()
    
    def create_noisy_images(self):
        """Create noisy versions of images for enhancement demonstrations"""
        np.random.seed(42)
        
        self.noisy_images = {}
        
        # Add Gaussian noise
        camera = self.images['camera'].astype(float)
        noise = np.random.normal(0, 15, camera.shape)
        self.noisy_images['gaussian'] = np.clip(camera + noise, 0, 255).astype(np.uint8)
        
        # Add salt and pepper noise
        salt_pepper = camera.copy()
        noise_mask = np.random.random(camera.shape)
        salt_pepper[noise_mask < 0.05] = 255  # Salt noise
        salt_pepper[noise_mask > 0.95] = 0    # Pepper noise
        self.noisy_images['salt_pepper'] = salt_pepper.astype(np.uint8)
        
        # Add motion blur
        motion_kernel = np.zeros((15, 15))
        motion_kernel[7, :] = 1
        motion_kernel = motion_kernel / 15
        self.noisy_images['motion_blur'] = ndimage.convolve(camera, motion_kernel).astype(np.uint8)
    
    def demonstrate_image_types(self):
        """Show different types of images and their characteristics"""
        print("=== IMAGE TYPES AND CHARACTERISTICS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Different image types
        image_types = [
            ('camera', 'Grayscale Natural Image'),
            ('coins', 'High Contrast Objects'),
            ('astronaut', 'Color Portrait'),
            ('checkerboard', 'Geometric Pattern'),
            ('text', 'Text Document'),
            ('coffee', 'Textured Natural Scene')
        ]
        
        for i, (img_name, title) in enumerate(image_types):
            row, col = i // 3, i % 3
            img = self.images[img_name]
            
            if len(img.shape) == 3:  # Color image
                axes[row, col].imshow(img)
            else:  # Grayscale
                axes[row, col].imshow(img, cmap='gray')
            
            axes[row, col].set_title(f'{title}\nShape: {img.shape}')
            axes[row, col].axis('off')
            
            # Print characteristics
            if len(img.shape) == 3:
                print(f"{img_name}: {img.shape}, Color, Range: [{img.min()}, {img.max()}]")
            else:
                print(f"{img_name}: {img.shape}, Grayscale, Range: [{img.min()}, {img.max()}]")
        
        plt.tight_layout()
        plt.show()
    
    def show_noise_types(self):
        """Demonstrate different types of image noise"""
        print("\n=== TYPES OF IMAGE NOISE ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(self.images['camera'], cmap='gray')
        axes[0, 0].set_title('Original Image')
        
        # Gaussian noise
        axes[0, 1].imshow(self.noisy_images['gaussian'], cmap='gray')
        axes[0, 1].set_title('Gaussian Noise')
        
        # Salt and pepper noise
        axes[1, 0].imshow(self.noisy_images['salt_pepper'], cmap='gray')
        axes[1, 0].set_title('Salt & Pepper Noise')
        
        # Motion blur
        axes[1, 1].imshow(self.noisy_images['motion_blur'], cmap='gray')
        axes[1, 1].set_title('Motion Blur')
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Noise characteristics:")
        print("• Gaussian: Random intensity variations, normally distributed")
        print("• Salt & Pepper: Random black and white pixels")
        print("• Motion Blur: Linear smearing due to camera or object movement")
        print("• Understanding noise types helps choose appropriate filtering methods")

# Demonstrate image processing introduction
img_demo = ImageProcessingDemo()
img_demo.demonstrate_image_types()
img_demo.show_noise_types()
```

## 2. Image Enhancement Techniques {#enhancement}

Image enhancement improves the visual quality of images by adjusting brightness, contrast, and reducing noise. The goal is to make images more suitable for human viewing or further automated analysis.

### Brightness and Contrast Adjustment

**Linear Transformation**: g(x,y) = a·f(x,y) + b
- a controls contrast (a > 1 increases contrast)
- b controls brightness (positive b increases brightness)

**Gamma Correction**: g(x,y) = c·f(x,y)^γ
- γ < 1 brightens the image
- γ > 1 darkens the image
- Useful for correcting display characteristics

### Histogram Equalization

Redistributes pixel intensities to achieve uniform histogram distribution, improving contrast in low-contrast images.

**Adaptive Histogram Equalization**: Applies equalization locally to small regions, preventing over-enhancement of already well-contrasted areas.

### Noise Reduction Techniques

**Linear Filtering**: Using convolution with smoothing kernels
**Non-linear Filtering**: Median filtering, bilateral filtering
**Advanced Methods**: Wiener filtering, wavelet denoising

```python
class ImageEnhancement:
    """Image enhancement techniques implementation"""
    
    def __init__(self):
        self.demo_image = data.camera()
        self.noisy_image = self.add_noise(self.demo_image)
    
    def add_noise(self, image):
        """Add mixed noise to image for demonstration"""
        noisy = image.astype(float)
        
        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, 20, image.shape)
        noisy += gaussian_noise
        
        # Add some impulse noise
        impulse_mask = np.random.random(image.shape) < 0.02
        noisy[impulse_mask] = np.random.choice([0, 255], size=np.sum(impulse_mask))
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def brightness_contrast_adjustment(self):
        """Demonstrate brightness and contrast adjustments"""
        print("=== BRIGHTNESS AND CONTRAST ADJUSTMENT ===")
        
        # Create different brightness/contrast versions
        adjustments = [
            (0.5, -50, 'Dark, Low Contrast'),
            (1.0, 0, 'Original'),
            (1.5, 30, 'Bright, High Contrast'),
            (2.0, -30, 'Very High Contrast')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (alpha, beta, title) in enumerate(adjustments):
            # Linear transformation: g(x,y) = alpha * f(x,y) + beta
            adjusted = np.clip(alpha * self.demo_image.astype(float) + beta, 0, 255).astype(np.uint8)
            
            axes[i].imshow(adjusted, cmap='gray')
            axes[i].set_title(f'{title}\nα={alpha}, β={beta}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def gamma_correction_demo(self):
        """Demonstrate gamma correction"""
        print("\n=== GAMMA CORRECTION ===")
        
        gamma_values = [0.5, 1.0, 1.5, 2.2]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, gamma in enumerate(gamma_values):
            # Normalize to [0,1], apply gamma, scale back to [0,255]
            normalized = self.demo_image / 255.0
            corrected = np.power(normalized, gamma)
            result = (corrected * 255).astype(np.uint8)
            
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f'γ = {gamma}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Gamma correction effects:")
        print("• γ < 1: Brightens dark areas, reduces contrast")
        print("• γ = 1: No change")
        print("• γ > 1: Darkens bright areas, increases contrast")
        print("• γ = 2.2: Standard monitor correction")
    
    def histogram_equalization_demo(self):
        """Demonstrate histogram equalization"""
        print("\n=== HISTOGRAM EQUALIZATION ===")
        
        # Original image
        original = self.demo_image
        
        # Global histogram equalization
        equalized = exposure.equalize_hist(original)
        equalized = (equalized * 255).astype(np.uint8)
        
        # Adaptive histogram equalization (CLAHE)
        clahe = exposure.equalize_adapthist(original, clip_limit=0.03)
        clahe = (clahe * 255).astype(np.uint8)
        
        # Create visualization
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        images = [original, equalized, clahe]
        titles = ['Original', 'Global Equalization', 'Adaptive (CLAHE)']
        
        for i, (img, title) in enumerate(zip(images, titles)):
            # Display image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(title)
            axes[i, 0].axis('off')
            
            # Display histogram
            axes[i, 1].hist(img.ravel(), bins=50, alpha=0.7, color='blue')
            axes[i, 1].set_title(f'{title} - Histogram')
            axes[i, 1].set_xlabel('Pixel Intensity')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def noise_reduction_techniques(self):
        """Demonstrate various noise reduction methods"""
        print("\n=== NOISE REDUCTION TECHNIQUES ===")
        
        # Apply different filtering techniques
        gaussian_filtered = filters.gaussian(self.noisy_image, sigma=1.0)
        median_filtered = filters.median(self.noisy_image, selem=morphology.disk(2))
        bilateral_filtered = filters.rank.mean_bilateral(
            self.noisy_image, 
            morphology.disk(3), 
            s0=10, s1=10
        )
        
        # Convert back to uint8
        gaussian_filtered = (gaussian_filtered * 255).astype(np.uint8) if gaussian_filtered.max() <= 1 else gaussian_filtered.astype(np.uint8)
        
        # Create comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        images = [
            (self.noisy_image, 'Noisy Original'),
            (gaussian_filtered, 'Gaussian Filter'),
            (median_filtered, 'Median Filter'),
            (bilateral_filtered, 'Bilateral Filter')
        ]
        
        for i, (img, title) in enumerate(images):
            row, col = i // 2, i % 2
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and display PSNR (Peak Signal-to-Noise Ratio)
        def calculate_psnr(original, filtered):
            mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            return psnr
        
        print("Noise reduction performance (higher PSNR is better):")
        filters_list = [
            (gaussian_filtered, 'Gaussian Filter'),
            (median_filtered, 'Median Filter'),
            (bilateral_filtered, 'Bilateral Filter')
        ]
        
        for filtered_img, name in filters_list:
            psnr = calculate_psnr(self.demo_image, filtered_img)
            print(f"{name}: PSNR = {psnr:.2f} dB")

# Demonstrate image enhancement
enhancement = ImageEnhancement()
enhancement.brightness_contrast_adjustment()
enhancement.gamma_correction_demo()
enhancement.histogram_equalization_demo()
enhancement.noise_reduction_techniques()
```

## 3. Spatial Domain Filtering {#spatial-filtering}

Spatial domain filtering applies mathematical operations directly to pixel neighborhoods using convolution with filter kernels. This fundamental technique enables edge detection, smoothing, sharpening, and feature extraction.

### Convolution Operation

The convolution operation combines the input image with a filter kernel:
**g(x,y) = Σᵢ Σⱼ h(i,j) × f(x+i, y+j)**

Where:
- f(x,y) is the input image
- h(i,j) is the filter kernel
- g(x,y) is the output image

### Types of Filters

**Low-pass Filters**: Reduce high-frequency components (noise, fine details)
- Gaussian filter: Smooth averaging with normal distribution weights
- Box filter: Simple averaging filter
- Bilateral filter: Edge-preserving smoothing

**High-pass Filters**: Enhance high-frequency components (edges, details)
- Laplacian filter: Second derivative edge detection
- Unsharp masking: Sharpening by subtracting blurred version

**Band-pass Filters**: Select specific frequency ranges

### Edge Detection Kernels

**Sobel Operators**: Detect edges using gradient approximation
**Prewitt Operators**: Similar to Sobel with different weights
**Roberts Cross-Gradient**: Simple diagonal edge detection
**Laplacian of Gaussian (LoG)**: Second derivative edge detection

```python
class SpatialFiltering:
    """Comprehensive spatial domain filtering demonstrations"""
    
    def __init__(self):
        self.test_image = data.camera()
        self.create_custom_kernels()
    
    def create_custom_kernels(self):
        """Create various filter kernels for demonstration"""
        self.kernels = {
            # Smoothing kernels
            'box_3x3': np.ones((3, 3)) / 9,
            'box_5x5': np.ones((5, 5)) / 25,
            'gaussian_3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            
            # Edge detection kernels
            'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'prewitt_y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            'roberts_x': np.array([[1, 0], [0, -1]]),
            'roberts_y': np.array([[0, 1], [-1, 0]]),
            'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            
            # Sharpening kernels
            'sharpen_basic': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            'sharpen_strong': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            
            # High-pass filters
            'high_pass': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        }
    
    def demonstrate_convolution_operation(self):
        """Show how convolution operation works step by step"""
        print("=== CONVOLUTION OPERATION DEMONSTRATION ===")
        
        # Create a simple 5x5 test image
        test_pattern = np.array([
            [0, 0, 0, 0, 0],
            [0, 50, 100, 50, 0],
            [0, 100, 255, 100, 0],
            [0, 50, 100, 50, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Simple 3x3 edge detection kernel
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        # Manual convolution for center pixel
        center_region = test_pattern[1:4, 1:4]
        convolution_result = np.sum(center_region * edge_kernel)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original pattern
        im1 = axes[0].imshow(test_pattern, cmap='gray', interpolation='nearest')
        axes[0].set_title('5×5 Test Pattern')
        axes[0].set_xticks(range(5))
        axes[0].set_yticks(range(5))
        plt.colorbar(im1, ax=axes[0])
        
        # Add grid and values
        for i in range(5):
            for j in range(5):
                axes[0].text(j, i, f'{test_pattern[i,j]}', ha='center', va='center',
                           color='white' if test_pattern[i,j] > 127 else 'black')
        
        # Edge kernel
        im2 = axes[1].imshow(edge_kernel, cmap='RdBu', interpolation='nearest')
        axes[1].set_title('3×3 Edge Detection Kernel')
        axes[1].set_xticks(range(3))
        axes[1].set_yticks(range(3))
        plt.colorbar(im2, ax=axes[1])
        
        for i in range(3):
            for j in range(3):
                axes[1].text(j, i, f'{edge_kernel[i,j]}', ha='center', va='center')
        
        # Convolution result
        full_result = ndimage.convolve(test_pattern.astype(float), edge_kernel, mode='constant')
        im3 = axes[2].imshow(full_result, cmap='RdYlBu', interpolation='nearest')
        axes[2].set_title(f'Convolution Result\nCenter: {convolution_result}')
        axes[2].set_xticks(range(5))
        axes[2].set_yticks(range(5))
        plt.colorbar(im3, ax=axes[2])
        
        for i in range(5):
            for j in range(5):
                axes[2].text(j, i, f'{full_result[i,j]:.0f}', ha='center', va='center',
                           color='white' if abs(full_result[i,j]) > 50 else 'black')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Manual convolution calculation for center pixel:")
        print(f"Region: \n{center_region}")
        print(f"Kernel: \n{edge_kernel}")
        print(f"Element-wise multiplication and sum: {convolution_result}")
    
    def compare_smoothing_filters(self):
        """Compare different smoothing filter effects"""
        print("\n=== SMOOTHING FILTERS COMPARISON ===")
        
        smoothing_filters = ['box_3x3', 'box_5x5', 'gaussian_3x3']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original image
        axes[0, 0].imshow(self.test_image, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Apply smoothing filters
        for i, filter_name in enumerate(smoothing_filters):
            kernel = self.kernels[filter_name]
            filtered = ndimage.convolve(self.test_image.astype(float), kernel)
            
            axes[0, i+1].imshow(filtered, cmap='gray')
            axes[0, i+1].set_title(f'{filter_name.replace("_", " ").title()}')
            axes[0, i+1].axis('off')
        
        # Show kernels
        for i, filter_name in enumerate(smoothing_filters):
            kernel = self.kernels[filter_name]
            
            # Pad kernel for visualization if needed
            display_kernel = kernel
            if kernel.shape[0] < 5:
                pad_size = (5 - kernel.shape[0]) // 2
                display_kernel = np.pad(kernel, pad_size, mode='constant', constant_values=0)
            
            im = axes[1, i+1].imshow(display_kernel, cmap='Blues', interpolation='nearest')
            axes[1, i+1].set_title(f'{filter_name.replace("_", " ").title()} Kernel')
            
            # Add values to kernel visualization
            for row in range(display_kernel.shape[0]):
                for col in range(display_kernel.shape[1]):
                    if display_kernel[row, col] != 0:
                        axes[1, i+1].text(col, row, f'{display_kernel[row,col]:.2f}', 
                                        ha='center', va='center', fontsize=8)
        
        axes[1, 0].axis('off')  # Empty space for alignment
        
        plt.tight_layout()
        plt.show()
    
    def edge_detection_comparison(self):
        """Compare different edge detection methods"""
        print("\n=== EDGE DETECTION COMPARISON ===")
        
        edge_methods = {
            'Sobel X': 'sobel_x',
            'Sobel Y': 'sobel_y',
            'Sobel Magnitude': None,  # Special case
            'Prewitt X': 'prewitt_x',
            'Prewitt Y': 'prewitt_y',
            'Laplacian': 'laplacian'
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Calculate Sobel magnitude
        sobel_x = ndimage.convolve(self.test_image.astype(float), self.kernels['sobel_x'])
        sobel_y = ndimage.convolve(self.test_image.astype(float), self.kernels['sobel_y'])
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        results = {}
        
        for i, (method_name, kernel_name) in enumerate(edge_methods.items()):
            if method_name == 'Sobel Magnitude':
                result = sobel_magnitude
            else:
                kernel = self.kernels[kernel_name]
                result = ndimage.convolve(self.test_image.astype(float), kernel)
            
            results[method_name] = result
            
            axes[i].imshow(np.abs(result), cmap='gray')
            axes[i].set_title(method_name)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show edge detection kernels
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        kernel_names = ['sobel_x', 'sobel_y', 'prewitt_x', 'prewitt_y', 'laplacian']
        
        for i, kernel_name in enumerate(kernel_names):
            kernel = self.kernels[kernel_name]
            im = axes[i].imshow(kernel, cmap='RdBu', interpolation='nearest', vmin=-2, vmax=2)
            axes[i].set_title(f'{kernel_name.replace("_", " ").title()} Kernel')
            
            # Add values
            for row in range(kernel.shape[0]):
                for col in range(kernel.shape[1]):
                    axes[i].text(col, row, f'{kernel[row,col]}', 
                               ha='center', va='center', fontsize=10, fontweight='bold')
        
        axes[5].axis('off')  # Hide unused subplot
        
        plt.tight_layout()
        plt.show()
    
    def sharpening_demonstration(self):
        """Demonstrate image sharpening techniques"""
        print("\n=== IMAGE SHARPENING TECHNIQUES ===")
        
        # Create a slightly blurred version for better demonstration
        blurred = filters.gaussian(self.test_image, sigma=1.5)
        
        # Apply different sharpening methods
        sharp_basic = ndimage.convolve(blurred, self.kernels['sharpen_basic'])
        sharp_strong = ndimage.convolve(blurred, self.kernels['sharpen_strong'])
        
        # Unsharp masking
        gaussian_blur = filters.gaussian(blurred, sigma=1.0)
        unsharp_mask = blurred + 1.5 * (blurred - gaussian_blur)
        
        # High-pass filter
        high_pass = ndimage.convolve(blurred, self.kernels['high_pass'])
        high_pass_enhanced = blurred + 0.5 * high_pass
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        images = [
            (self.test_image, 'Original'),
            (blurred, 'Blurred'),
            (sharp_basic, 'Basic Sharpening'),
            (sharp_strong, 'Strong Sharpening'),
            (unsharp_mask, 'Unsharp Masking'),
            (high_pass_enhanced, 'High-pass Enhanced')
        ]
        
        for i, (img, title) in enumerate(images):
            row, col = i // 3, i % 3
            # Clip values to valid range
            display_img = np.clip(img, 0, 255)
            axes[row, col].imshow(display_img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Sharpening method characteristics:")
        print("• Basic Sharpening: Simple kernel-based enhancement")
        print("• Strong Sharpening: More aggressive enhancement, may create artifacts")
        print("• Unsharp Masking: Subtracts blurred version from original")
        print("• High-pass Enhanced: Adds high-frequency components back to image")

# Demonstrate spatial filtering
spatial_filter = SpatialFiltering()
spatial_filter.demonstrate_convolution_operation()
spatial_filter.compare_smoothing_filters()
spatial_filter.edge_detection_comparison()
spatial_filter.sharpening_demonstration()
```

## Summary

This chapter covered essential image processing techniques:

### Key Concepts:
1. **Image Processing Pipeline**: From acquisition to analysis
2. **Enhancement Techniques**: Brightness, contrast, noise reduction
3. **Spatial Filtering**: Convolution operations for various effects
4. **Edge Detection**: Gradient-based methods for feature extraction

### Practical Applications:
- **Medical Imaging**: Enhancing diagnostic images
- **Industrial Inspection**: Quality control and defect detection
- **Digital Photography**: Professional image editing
- **Security Systems**: Image improvement for recognition

### Best Practices:
- Choose appropriate filters based on noise characteristics
- Consider computational efficiency for real-time applications
- Validate enhancement results with domain experts
- Combine multiple techniques for optimal results
- Preserve important image features during processing

---

## Exercises

1. **Custom Filter Design**: Create specialized filters for specific applications
2. **Noise Analysis**: Develop methods to characterize and remove specific noise types
3. **Real-time Processing**: Implement efficient algorithms for video processing
4. **Quality Metrics**: Develop objective measures for enhancement evaluation
5. **Application Integration**: Build complete image processing pipelines

---

*Image processing techniques are fundamental tools for preparing visual data for analysis. Understanding these methods enables effective preprocessing for computer vision applications.* 