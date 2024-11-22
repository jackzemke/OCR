import cv2
from matplotlib import pyplot as plt


# To begin the process, we're gonna import the input image and do some pre processing to make
#it easier to extract the characters and such 

# For now, i'm just going to use a photo of a passage in a course syllabus. this will be a good starting 
# point because it is harder to process than a screenshot of a document, but easier than handwritten notes 
# (the final goal)

pic = "far.jpg"

def gray(image): #convert to grayscale to reduce computational overhead
    return cv2.imread(image,0)

def denoise_image(image, method='gaussian'):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        raise ValueError("Invalid method. Choose 'gaussian', 'median', or 'bilateral'.")
    
if __name__ == "__main__":
    image_path = pic
    gray_image = gray(image_path)  # From previous step
    cv2.imwrite('gray.jpg', gray_image)

    # Apply Gaussian blur
    # gaussian_denoised = denoise_image(gray_image, method='gaussian')
    # cv2.imwrite('Gaussian.jpg', gaussian_denoised)

    # # Apply Median blur
    # median_denoised = denoise_image(gray_image, method='median')
    # cv2.imwrite('Median.jpg', median_denoised)

    # # Apply Bilateral filter
    # bilateral_denoised = denoise_image(gray_image, method='bilateral')
    # cv2.imwrite('Bilateral.jpg', bilateral_denoised)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()