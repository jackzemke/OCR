import cv2
from matplotlib import pyplot as plt
import numpy as np


# To begin the process, we're gonna import the input image and do some pre processing to make
#it easier to extract the characters and such 

# For now, i'm just going to use a photo of a passage in a course syllabus. this will be a good starting 
# point because it is harder to process than a screenshot of a document, but easier than handwritten notes 
# (the final goal)

# pic = "far.jpg"
# pic = 'fulltext.JPG'
pic = 'full2.JPG'
# pic = 'math.jpg'

def gray(image): #convert to grayscale to reduce computational overhead
    return cv2.imread(image,0)

def blur(image):
    return cv2.medianBlur(image, 5)

def binarize_image(image, method='global'):
    if method == 'global':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == 'adaptive':
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'inverted':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(binary)
    else:
        raise ValueError("Invalid method. Choose 'global', 'adaptive', or 'inverted'.")

def segment_lines(binary_image):
    min_line_height = 10
    padding = 9
    print("[INFO] Starting line segmentation with padding...")

    # Compute horizontal projection profile: Sum white pixels (255)
    horizontal_projection = np.sum(binary_image == 255, axis=1)
    print(f"[DEBUG] Horizontal projection profile computed. Length: {len(horizontal_projection)}")

    # Set a threshold to detect lines
    threshold = np.max(horizontal_projection) * 0.065  # 10% of the maximum white pixel count
    print(f"[DEBUG] Threshold for line detection: {threshold}")

    # Detect line boundaries
    line_indices = []
    in_line = False
    for i, pixel_sum in enumerate(horizontal_projection):
        if pixel_sum > threshold and not in_line:
            # Starting a new line
            line_start = i
            in_line = True
        elif pixel_sum <= threshold and in_line:
            # Ending the current line
            line_end = i
            if line_end - line_start >= min_line_height:  # Ignore small lines
                # Add padding
                line_start = max(0, line_start - padding)
                line_end = min(binary_image.shape[0], line_end + padding)
                line_indices.append((line_start, line_end))
                print(f"[DEBUG] Line detected from row {line_start} to {line_end}")
            in_line = False

    # Extract lines
    print(f"[INFO] Total lines detected: {len(line_indices)}")
    lines = [binary_image[start:end, :] for start, end in line_indices]
    return lines, line_indices

def segment_words_dynamic(line_image, min_word_width=22, gap_percentage=0.01):
    """
    Segments a single line into individual words based on the vertical projection profile,
    using a percentage-based dynamic threshold for gaps.
    Args:
        line_image: Binary image of a single text line.
        min_word_width: Minimum width (in pixels) for a valid word.
        gap_percentage: Percentage of the maximum vertical projection to define a gap threshold.
    Returns:
        words: List of binary images, each corresponding to an individual word.
        word_indices: List of (start_col, end_col) tuples for each word.
    """
    print("[INFO] Starting word segmentation with dynamic threshold...")

    # Compute vertical projection profile
    vertical_projection = np.sum(line_image == 255, axis=0)
    max_pixel_sum = np.max(vertical_projection)
    gap_threshold = max_pixel_sum * gap_percentage  # Dynamic threshold for gaps
    print(f"[DEBUG] Maximum vertical projection: {max_pixel_sum}")
    print(f"[DEBUG] Gap threshold (percentage-based): {gap_threshold}")

    # Detect word boundaries
    word_indices = []
    in_word = False
    for col, pixel_sum in enumerate(vertical_projection):
        if pixel_sum > gap_threshold and not in_word:
            # Starting a new word
            word_start = col
            in_word = True
        elif pixel_sum <= gap_threshold and in_word:
            # Ending the current word
            word_end = col
            if word_end - word_start >= min_word_width:  # Ignore small artifacts
                word_indices.append((word_start, word_end))
                print(f"[DEBUG] Word detected from column {word_start} to {word_end}")
            in_word = False

    # Extract words
    words = [line_image[:, start:end] for start, end in word_indices]
    print(f"[INFO] Total words detected: {len(word_indices)}")

    return words, word_indices

def segment_words(line_image, min_word_width=10, gap_threshold=5):
    """
    Segments a single line into individual words based on the vertical projection profile.

    Args:
        line_image: Binary image of a single text line (white text on black background).
        min_word_width: Minimum width (in pixels) for a valid word.
        gap_threshold: Minimum gap (in pixels) to detect word boundaries.

    Returns:
        words: List of binary images, each corresponding to an individual word.
        word_indices: List of (start_col, end_col) tuples for each word.
    """
    print("[INFO] Starting word segmentation...")

    # Compute vertical projection profile (sum white pixels in each column)
    vertical_projection = np.sum(line_image == 255, axis=0)
    print(f"[DEBUG] Vertical projection profile computed. Length: {len(vertical_projection)}")

    # Detect word boundaries
    word_indices = []
    in_word = False
    for col, pixel_sum in enumerate(vertical_projection):
        if pixel_sum > gap_threshold and not in_word:
            # Starting a new word
            word_start = col
            in_word = True
        elif pixel_sum <= gap_threshold and in_word:
            # Ending the current word
            word_end = col
            if word_end - word_start >= min_word_width:  # Ignore small artifacts
                word_indices.append((word_start, word_end))
                print(f"[DEBUG] Word detected from column {word_start} to {word_end}")
            in_word = False

    # Extract words
    words = [line_image[:, start:end] for start, end in word_indices]
    print(f"[INFO] Total words detected: {len(word_indices)}")

    return words, word_indices

import matplotlib.pyplot as plt

def plot_vertical_projection(line_image):
    vertical_projection = np.sum(line_image == 255, axis=0)
    plt.plot(vertical_projection)
    plt.title("Vertical Projection Profile")
    plt.xlabel("Column Index")
    plt.ylabel("White Pixel Count")
    plt.show()

def visualize_lines(lines, binary_image):
    for idx, line in enumerate(lines):
        cv2.imshow(f'Line {idx + 1}', line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_horizontal_projection(binary_image):
    horizontal_projection = np.sum(binary_image == 0, axis=1)
    plt.plot(horizontal_projection)
    plt.title("Horizontal Projection Profile")
    plt.xlabel("Row Index")
    plt.ylabel("Black Pixel Count")
    plt.show()

def segment_words_with_components(line_image):
    """
    Segments words using connected component analysis.
    """
    print("[INFO] Starting word segmentation with connected components...")

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    # Extract bounding boxes for each connected component
    word_indices = []
    words = []
    for i in range(1, num_labels):  # Skip the background (label 0)
        x, y, w, h, area = stats[i]
        if w > 10:  # Minimum width for a valid word
            word_indices.append((x, x + w))
            word = line_image[:, x:x + w]  # Extract the word
            words.append(word)
            print(f"[DEBUG] Word detected at column {x} to {x + w} with width {w}")

    print(f"[INFO] Total words detected: {len(words)}")
    return words, word_indices

def render_words_side_by_side(line_image, words, line_number):
    """
    Renders all words in a single window, displayed side by side with neon green lines between words.
    """
    print(f"[INFO] Rendering words side by side for Line {line_number}...")
    
    # Calculate the total width needed to display all words side by side
    total_width = sum(word.shape[1] for word in words) + len(words)  # Add 1px gaps for the green lines
    max_height = max(word.shape[0] for word in words)

    # Create a blank image to hold all words side by side
    canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)  # Use 3 channels for color
    
    # Place each word onto the canvas with green lines separating them
    current_x = 0
    for word in words:
        # Add the word to the canvas (convert grayscale to 3-channel for compatibility)
        word_rgb = cv2.cvtColor(word, cv2.COLOR_GRAY2BGR)
        canvas[0:word.shape[0], current_x:current_x + word.shape[1]] = word_rgb
        
        # Draw a neon green line to mark the boundary
        current_x += word.shape[1]
        if current_x < canvas.shape[1]:  # Ensure we're within the canvas bounds
            canvas[:, current_x:current_x + 1] = (0, 255, 0)  # Neon green line
            current_x += 1  # Move past the line

    # Show the full line and the side-by-side words
    cv2.imshow(f"Line {line_number} - Full Line", line_image)
    cv2.imshow(f"Line {line_number} - Words Side by Side with Boundaries", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segment_words_by_spaces(line_image, space_threshold=5, min_space_width=10, min_word_width=20):
    """
    Segments words by identifying large black spaces in the vertical projection profile.
    
    Args:
        line_image: Binary image of a single text line.
        space_threshold: Maximum number of white pixels in a column to consider it "black".
        min_space_width: Minimum width of consecutive black columns to qualify as a space.
        min_word_width: Minimum width (in pixels) for a valid word.

    Returns:
        words: List of binary images for each word.
        word_indices: List of (start_col, end_col) tuples for each word.
    """
    print("[INFO] Starting word segmentation by spaces...")

    # Compute vertical projection profile
    vertical_projection = np.sum(line_image == 255, axis=0)
    print(f"[DEBUG] Vertical projection profile calculated. Length: {len(vertical_projection)}")

    # Identify space regions (clusters of black-heavy columns)
    space_indices = []
    in_space = False
    for col, pixel_sum in enumerate(vertical_projection):
        if pixel_sum <= space_threshold and not in_space:
            # Start of a space
            space_start = col
            in_space = True
        elif pixel_sum > space_threshold and in_space:
            # End of a space
            space_end = col
            if space_end - space_start >= min_space_width:
                space_indices.append((space_start, space_end))
            in_space = False

    print(f"[DEBUG] Spaces identified: {space_indices}")

    # Use spaces to break the line into words
    word_indices = []
    prev_space_end = 0
    for space_start, space_end in space_indices:
        # Define the word region as the space between the previous space and the current space
        if space_start - prev_space_end >= min_word_width:
            word_indices.append((prev_space_end, space_start))
        prev_space_end = space_end

    # Add the final word after the last space
    if line_image.shape[1] - prev_space_end >= min_word_width:
        word_indices.append((prev_space_end, line_image.shape[1]))

    print(f"[DEBUG] Word regions identified: {word_indices}")

    # Extract word images
    words = [line_image[:, start:end] for start, end in word_indices]
    print(f"[INFO] Total words detected: {len(words)}")

    return words, word_indices

def preprocess_line_for_words(line_image):
    """
    Preprocess the line image for word segmentation by applying horizontal dilation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Wider horizontal dilation
    dilated = cv2.dilate(line_image, kernel, iterations=1)
    return dilated

if __name__ == "__main__":
    # Load your binarized image
    image_path = pic
    gray_image = gray(image_path)
    denoised_image = blur(gray_image)
    binary_image = binarize_image(denoised_image, method='inverted')

    cv2.imwrite('debug_binary_image.jpg', binary_image)

    # Perform line segmentation
    lines, line_indices = segment_lines(binary_image)
    count = 0
    for idx, line in enumerate(lines):
        count += 1
        if count > 2:
            break
        line = preprocess_line_for_words(line)
        # plot_vertical_projection(line)

        print(f"[INFO] Processing Line {idx + 1}...")
        
        # Segment words in the line
        words, word_indices = segment_words_by_spaces(line, space_threshold=5, min_space_width=80, min_word_width=20)

        # words, word_indices = segment_words_dynamic(line)
        # words, word_indices = segment_words_with_components(line)

        if len(word_indices) != 0:
            for word_idx, word in enumerate(words):
                cv2.imshow(f'Line {idx + 1} - Word {word_idx + 1}', word)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            render_words_side_by_side(line, words, idx + 1)

        
        
