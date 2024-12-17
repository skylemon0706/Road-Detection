import itertools
import cv2
import math
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

def dark_channel(image, size=15):
    """Calculate the dark channel prior of an image."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """Estimate atmospheric light based on the dark channel."""
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest_pixels = int(max(num_pixels // 1000, 1))
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[-num_brightest_pixels:], dark_channel.shape)
    A = np.mean(image[indices], axis=0)
    return A

def defog_image(image):
    """Defog the input image using improved dark channel prior."""
    I = image.astype(float)
    dark_channel_img = dark_channel(I)
    A = estimate_atmospheric_light(I, dark_channel_img)
    omega = 0.95
    t = 1 - omega * (dark_channel_img / np.max(A))
    t = np.clip(t, 0.1, 1)
    t = cv2.bilateralFilter(t.astype(np.float32), 5, 0.1, 0.1)
    J = np.empty_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A[i]) / t + A[i]
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J, dark_channel_img, t

def detect_lines(cropped_edges, image, min_distance=100):
    """Detect lines, filter, sort, and reflect lines using an average axis if needed."""
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=60, maxLineGap=300)
    line_image = np.zeros_like(image)
    left_line_image = np.zeros_like(image)
    right_line_image = np.zeros_like(image)
    filtered_lines = []
    
    # Image dimensions
    height, width = image.shape[:2]
    mid_x = width // 2

    if lines is not None:
        # Filter and deduplicate lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
            else:
                slope = np.inf
                intercept = x1
            
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Deduplicate lines
            keep = True
            for i, (f_slope, f_intercept, f_length, f_coords) in enumerate(filtered_lines):
                if abs(f_slope - slope) < 0.3:
                    dist = abs(f_intercept - intercept)
                    if dist < min_distance:
                        if length > f_length:
                            filtered_lines[i] = (slope, intercept, length, (x1, y1, x2, y2))
                        keep = False
                        break

            if keep and abs(slope) > 0.35:  # Ignore nearly vertical lines
                filtered_lines.append((slope, intercept, length, (x1, y1, x2, y2)))
        
        # Strictly categorize left and right lines
        left_lines = []
        right_lines = []
        for slope, intercept, length, (x1, y1, x2, y2) in filtered_lines:
            avg_x = (x1 + x2) / 2
            if avg_x < mid_x and slope < 0:  # Left lines
                left_lines.append((slope, intercept, length, (x1, y1, x2, y2)))
            elif avg_x > mid_x and slope > 0:  # Right lines
                right_lines.append((slope, intercept, length, (x1, y1, x2, y2)))

        # Filter lines based on 60th percentile length
        def filter_lines_by_percentile(lines, percentile=60):
            if not lines:
                return []
            lengths = [line[2] for line in lines]
            threshold_length = np.percentile(lengths, percentile)
            return [line for line in lines if line[2] >= threshold_length]

        left_lines = filter_lines_by_percentile(left_lines, 60)
        right_lines = filter_lines_by_percentile(right_lines, 60)

        # Draw left and right lines
        for _, _, _, (x1, y1, x2, y2) in left_lines:
            cv2.line(left_line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        for _, _, _, (x1, y1, x2, y2) in right_lines:
            cv2.line(right_line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # Calculate the average axis from filtered lines
        if filtered_lines:
            average_axis = int(np.mean([(x1 + x2) / 2 for _, _, _, (x1, _, x2, _) in filtered_lines]))
        else:
            average_axis = mid_x  # Default to mid_x if no lines

        # Reflect a line based on average axis
        def reflect_line(x1, y1, x2, y2, axis):
            x1_reflected = 2 * axis - x1
            x2_reflected = 2 * axis - x2
            return x1_reflected, y1, x2_reflected, y2

        # Handle missing lines with reflections
        if not left_lines and right_lines:
            # Reflect the longest right line
            _, _, _, (x1, y1, x2, y2) = max(right_lines, key=lambda l: l[2])
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            reflected = reflect_line(x1, y1, x2, y2, average_axis)
            cv2.line(line_image, (reflected[0], reflected[1]), (reflected[2], reflected[3]), (0, 255, 0), 5)

        elif not right_lines and left_lines:
            # Reflect the longest left line
            _, _, _, (x1, y1, x2, y2) = max(left_lines, key=lambda l: l[2])
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            reflected = reflect_line(x1, y1, x2, y2, average_axis)
            cv2.line(line_image, (reflected[0], reflected[1]), (reflected[2], reflected[3]), (0, 255, 0), 5)

        # Draw optimal pair of lines if both sets are available
        elif left_lines and right_lines:
            def calculate_area(line1, line2):
                _, _, _, (x1, y1, x2, y2) = line1
                _, _, _, (x3, y3, x4, y4) = line2
                return abs((x2 - x1) * (y4 - y3) - (x3 - x4) * (y2 - y1))

            min_area = float('inf')
            best_left_line = None
            best_right_line = None
            for left_line, right_line in itertools.product(left_lines, right_lines):
                area = calculate_area(left_line, right_line)
                if area < min_area:
                    min_area = area
                    best_left_line = left_line
                    best_right_line = right_line

            if best_left_line and best_right_line:
                _, _, _, (x1, y1, x2, y2) = best_left_line
                _, _, _, (x3, y3, x4, y4) = best_right_line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.line(line_image, (x3, y3), (x4, y4), (0, 255, 0), 5)

    return line_image, left_line_image, right_line_image







def detect_lanes(image):
    """Detect lanes in a defogged image while filtering out vertical lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([
        (0, height),
        (width, height),
        (width, height // 1.7),
        (0, height // 1.7)
    ], np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    line_image, left_line_image, right_line_image = detect_lines(cropped_edges, image)
    lane_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return lane_image, left_line_image, right_line_image, gray, blur, edges, cropped_edges


# Streamlit integration
st.title("Foggy Lane Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    input_image = np.array(Image.open(uploaded_file))
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    # Step 1: Defog the image
    defogged_image, dark_channel_img, transmission_map = defog_image(input_image)
    st.image(defogged_image, caption="Defogged Image", use_container_width=True)
    st.image(dark_channel_img, caption="Dark Channel", use_container_width=True, clamp=True)
    st.image(transmission_map, caption="Transmission Map", use_container_width=True, clamp=True)

    # Step 2: Detect lanes
    lane_image, left_line_image, right_line_image, gray, blur, edges, cropped_edges = detect_lanes(defogged_image)
    st.image(gray, caption="Grayscale Image", use_container_width=True, clamp=True)
    st.image(blur, caption="Blurred Image", use_container_width=True, clamp=True)
    st.image(edges, caption="Edges", use_container_width=True, clamp=True)
    st.image(cropped_edges, caption="Region of Interest", use_container_width=True, clamp=True)

    # New: Display left and right line images
    st.image(left_line_image, caption="Left Lines", use_container_width=True, clamp=True)
    st.image(right_line_image, caption="Right Lines", use_container_width=True, clamp=True)

    st.image(lane_image, caption="Final Lane Detection", use_container_width=True)

    # Download button
    lane_image_pil = Image.fromarray(lane_image.astype('uint8'))
    buffer = BytesIO()
    lane_image_pil.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Final Lane Detection Image",
        data=buffer,
        file_name="lane_detection_result.png",
        mime="image/png"
    )

