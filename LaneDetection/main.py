import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_synthetic_image():
    # Create a black image (height=540, width=960, 3 channels)
    height, width = 540, 960
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define color for markings (White)
    color = (255, 255, 255)
    thickness = 10

    # Draw left lane line (hypothetical)
    # Starting from bottom-left to center-ish
    cv2.line(image, (200, height), (400, 300), color, thickness)

    # Draw right lane line
    # Starting from bottom-right to center-ish
    cv2.line(image, (800, height), (560, 300), color, thickness)
    
    # Save the image
    cv2.imwrite('test_image.jpg', image)
    print("Created 'test_image.jpg' for testing.")
    return image


def preprocess_image(image):
    """
    Step 2: Preprocessing
    Applies Grayscale conversion and Gaussian Blur.
    """
    # 1. Convert to Grayscale
    # We convert from BGR (OpenCV default) to RGB for matplotlib usually, 
    # but for processing we need Gray.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur
    # Kernel size (5, 5) is standard. Must be odd numbers.
    # Deviation 0 means calculated from kernel size.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blur


def canny_edge_detector(image):
    """
    Step 3: Canny Edge Detection
    Detects strong edges in the image.
    """
    # 50 is the low threshold, 150 is the high threshold
    # Ratios of 1:2 or 1:3 are recommended.
    canny = cv2.Canny(image, 50, 150)
    return canny


def region_of_interest(image):
    """
    Step 4: Region of Interest (ROI)
    Masks the image to focus only on the road area.
    """
    height = image.shape[0]
    
    # Define a polygon (triangle) that covers the expected lane area
    # (Bottom Left, Bottom Right, Top Center)
    # These coordinates are specific to our synthetic image size (540, 960)
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    
    # Create black mask with same dimensions as image
    mask = np.zeros_like(image)
    
    # Fill our defined polygon with white color (255) on the mask
    cv2.fillPoly(mask, triangle, 255)
    
    # Perform bitwise AND operation
    # This keeps only the parts of the image where the mask is white
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image




def make_coordinates(image, line_parameters):
    """
    Step 6 Helper: Converts slope and intercept back into coordinates.
    """
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None # Handle case where no line is detected
        
    y1 = image.shape[0] # Bottom of the image
    y2 = int(y1 * (3/5)) # Go up to 3/5ths of the image height
    
    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Step 6: Optimization
    Calculates the average slope and intercept for left and right lanes.
    """
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
        
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Fit a polynomial of degree 1 (a line) -> returns [slope, intercept]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Determine if left or right lane based on slope
        # Negative slope = Left Lane (in image coordinates)
        # Positive slope = Right Lane
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    # Calculate average and return coordinates
    # axis=0 averages columns (slope with slope, int with int)
    
    lines_to_draw = []
    
    if len(left_fit) > 0:
        left_avg = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_avg)
        if left_line is not None:
            lines_to_draw.append([left_line])
            
    if len(right_fit) > 0:
        right_avg = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_avg)
        if right_line is not None:
            lines_to_draw.append([right_line])
            
    return np.array(lines_to_draw)

def display_lines(image, lines):
    """
    Step 5 Helper: Draws lines on a black image.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Reshape line from 2D array to 1D
            x1, y1, x2, y2 = line.reshape(4)
            # Draw line: image, start_point, end_point, color (Blue), thickness
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def create_synthetic_video(filename='test_video.avi'):
    """
    Generates a synthetic video of a road with moving lane lines.
    """
    height, width = 540, 960
    fps = 30
    duration = 5 # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    print(f"Generating synthetic video '{filename}'...")
    
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color = (255, 255, 255)
        thickness = 10
        
        # Simulate movement by shifting the top point of the lines
        shift = int(50 * np.sin(i / 10)) # Oscillation
        
        # Left Lane
        cv2.line(frame, (200, height), (400 + shift, 300), color, thickness)
        
        # Right Lane
        cv2.line(frame, (800, height), (560 + shift, 300), color, thickness)
        
        out.write(frame)
        
    out.release()
    print("Video generation complete.")

def process_frame(image):
    """
    Runs the entire lane detection pipeline on a single frame.
    """
    # 1. Preprocess
    # processed_image = preprocess_image(image) # Optional if Canny handles it well, but good practice
    # For synthetic moving video, we can skip complex preprocess or keep it simple
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Edge Detection
    canny_image = canny_edge_detector(blur)
    
    # 3. ROI
    cropped_image = region_of_interest(canny_image)
    
    # 4. Line Detection
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    # 5. Optimization
    averaged_lines = average_slope_intercept(image, lines)
    
    # 6. Visualization
    line_image = display_lines(image, averaged_lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    return combo_image

def main():
    print("OpenCV version:", cv2.__version__)
    
    # Step 7: Video Pipeline
    video_filename = 'test_video.avi'
    create_synthetic_video(video_filename)
    
    cap = cv2.VideoCapture(video_filename)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    print("Playing video with Lane Detection... (Press 'q' to quit)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Process the frame
            result_frame = process_frame(frame)
            
            # Display
            cv2.imshow('Lane Detection Result', result_frame)
            
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Step 7 Complete! Video processing finished.")

if __name__ == "__main__":
    main()
