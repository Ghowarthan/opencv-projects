                   # Lane Detection Project - Step by Step Learning
                   

This project is a step-by-step implementation of a Lane Detection system for autonomous vehicles using Python and OpenCV.

## Current Progress
- [x] **Step 1: Environment Setup** - Installing libraries and verifying setup.
- [x] **Step 2: Image Loading & Preprocessing** - Grayscale and Blur.
- [x] **Step 3: Edge Detection** - Canny Algorithm.
- [x] **Step 4: Region of Interest** - Masking the road area.
- [x] **Step 5: Line Detection** - Hough Transform.
- [x] **Step 6: Optimization** - Averaging lines.
- [x] **Step 7: Video Pipeline** - Processing video frames.

                     ## How to Run
      
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`
LANE DETECTION FOR AUTONOMOUS VEHICLES - STEP-BY-STEP TUTORIAL

================================================================================
1. PROJECT OVERVIEW
================================================================================
This project implements a lane detection system using computer vision techniques.
The goal is to identify lane markings on a road and highlight them. This is a 
fundamental task for self-driving cars to keep the vehicle centered in the lane.

Technologies Used:
- Python (Programming Language)
- OpenCV (Computer Vision Library)
- NumPy (Numerical processing for matrices/images)
- Matplotlib (Visualization)

================================================================================
2. STEP-BY-STEP IMPLEMENTATION PROCESS
================================================================================

--------------------------------------------------------------------------------
STEP 1: PREPARATION & SETUP
--------------------------------------------------------------------------------
Before processing real images, we need an environment and a test subject.
- We installed `opencv-python`, `numpy`, and `matplotlib`.
- We created a `main.py` script.
- We wrote a `create_synthetic_image()` function to generate a black image 
  with white lines. This ensures we have a perfect test case before dealing 
  with real-world noise.

--------------------------------------------------------------------------------
STEP 2: PREPROCESSING (Grayscale & Blur)
--------------------------------------------------------------------------------
Concept:
- Images are made of pixels. A standard image has 3 channels: Red, Green, Blue.
- Processing 3 channels is computationally expensive and color is often 
  distracting for edge detection (shadows, different paint colors).
- Noise (grainy spots) creates false edges.

Action:
1. Grayscale: Converted the image to black & white (1 channel).
   Code: `gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
2. Gaussian Blur: Smoothed the image to reduce noise using a 5x5 kernel.
   Code: `blur = cv2.GaussianBlur(gray, (5, 5), 0)`

--------------------------------------------------------------------------------
STEP 3: EDGE DETECTION (Canny)
--------------------------------------------------------------------------------
Concept:
- An "edge" is where the brightness of an image changes sharply (e.g., from 
  dark road to white paint).
- The Canny algorithm calculates the gradient (change) in all directions.

Action:
- We applied the Canny Edge Detector. 
- Thresholds (50, 150): Define what counts as a "strong" edge vs a "weak" edge.
  Code: `canny = cv2.Canny(image, 50, 150)`
- Result: An all-black image with thin white lines tracing the shapes.

--------------------------------------------------------------------------------
STEP 4: REGION OF INTEREST (ROI)
--------------------------------------------------------------------------------
Concept:
- The camera sees the sky, trees, and other cars. We only care about the road.
- The road is mathematically almost always in the bottom half of the frame in 
  a triangular shape converging towards the horizon.

Action:
- Defined a triangle polygon (coordinates dependent on image resolution).
- Generated a black mask and filled the triangle with white.
- Used "Bitwise AND" to combine the Canny image and the Mask.
- Result: Everything outside the triangle became black; only road edges remained.

--------------------------------------------------------------------------------
STEP 5: LINE DETECTION (Hough Transform)
--------------------------------------------------------------------------------
Concept:
- The computer sees edges as individual pixels, not "lines".
- The Hough Transform is a voting system. It checks every edge pixel/dot and 
  calculates: "If a line went through this dot, what angle/position would it have?"
- If enough dots vote for the same angle/position, it is considered a Line.

Action:
- Used `cv2.HoughLinesP`.
- Tuned parameters (threshold=100 votes, minLineLength=40px, maxLineGap=5px).
- Result: A list of start/end coordinates [(x1, y1, x2, y2)] for many small line segments.

--------------------------------------------------------------------------------
STEP 6: OPTIMIZATION (Smoothing)
--------------------------------------------------------------------------------
Concept:
- Step 5 gives us many choppy little lines (e.g., dashed lane markings).
- We want TWO solid lines: one Left Lane, one Right Lane.
- We need to average the math.

Action:
1. Calculate Slope (m) and Intercept (b) for every line segment: y = mx + b.
2. Filter: 
   - Negative slope = Left Lane (in image coordinates y increases downwards).
   - Positive slope = Right Lane.
3. Average: Took the average of all left slopes/intercepts and right slopes/intercepts.
4. Extrapolate: Calculated new x/y coordinates to draw lines from the bottom 
   of the screen up to the horizon (approx. middle of screen).

--------------------------------------------------------------------------------
STEP 7: VIDEO PIPELINE
--------------------------------------------------------------------------------
Concept:
- A video is just a series of images (frames) played fast.
- We wrap our logic into a function `process_frame()`.

Action:
- Created a loop that opens a video file (or webcam).
- Reads one frame.
- Sends frame to `process_frame()` -> Returns image with blue lines drawn.
- Displays the result.
- Repeats until video ends.

================================================================================
HOW TO RUN
================================================================================
1. Open terminal in the project folder.
2. Run: `python main.py`
3. The script will generate a 'test_video.avi' and play it with detection active.
4. Press 'q' to quit the window.

