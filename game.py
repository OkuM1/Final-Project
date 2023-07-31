import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
import numpy as np
import os
from datetime import datetime, timedelta

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Detects 1 Hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def rotate_frame(frame, angle):
    center = tuple(np.array(frame.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_frame

def draw_timer(screen, time_left, window_width):
    font = pygame.font.Font(None, 30)
    text = font.render("Time Left: {}s".format(time_left), True, (255, 255, 255))
    screen.blit(text, (window_width - 150, 10))

def main():
    cap = cv2.VideoCapture(0)

    pygame.init()

    # Get the resolution of the webcam feed
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the desired window size for pygame
    window_width = 800
    window_height = 600

    # Calculate the position to center the webcam feed
    x_offset = (window_width - webcam_width) // 2
    y_offset = (window_height - webcam_height) // 2

    screen = pygame.display.set_mode((window_width, window_height), RESIZABLE)
    pygame.display.set_caption("Hand Detection")

    clock = pygame.time.Clock()

    angle = 90  # Set the initial rotation angle to 90 degrees
    fullscreen = False

    pencil_mode = True  # Start with pencil mode
    drawing = False     # Flag to indicate if we're actively drawing

    # Create a surface to draw the pencil lines
    pencil_surface = pygame.Surface((webcam_width, webcam_height), pygame.SRCALPHA)

    # List to store drawing points for continuous line
    drawing_points = []

    # Start Screen
    start_screen = True
    player_name = ""

    # End Screen
    end_screen = False
    end_screen_time = 5
    timer_start_time = None
    retry_button = pygame.Rect(300, 300, 200, 50)

    # Game timer
    game_timer_start = None
    game_duration = timedelta(seconds=15)
    
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN:
                if start_screen:
                    if 650 <= event.pos[0] <= 790 and 10 <= event.pos[1] <= 50:
                        if fullscreen:
                            pygame.display.set_mode((window_width, window_height), RESIZABLE)
                        else:
                            pygame.display.set_mode((window_width, window_height), FULLSCREEN)
                        fullscreen = not fullscreen
                    elif 300 <= event.pos[0] <= 500 and 300 <= event.pos[1] <= 350:
                        start_screen = False
                        game_timer_start = datetime.now()
                elif end_screen:
                    if retry_button.collidepoint(event.pos):
                        end_screen = False
                        start_screen = True
                        player_name = ""
                        drawing_points = []
                        pencil_surface.fill((0, 0, 0, 0))
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
            elif event.type == KEYDOWN:
                if start_screen:
                    if event.key == K_RETURN:
                        start_screen = False
                        game_timer_start = datetime.now()
                    elif event.key == K_BACKSPACE:
                        player_name = player_name[:-1]
                    else:
                        player_name += event.unicode

        if start_screen:
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 50)
            text = font.render("Welcome to the Game!", True, (255, 255, 255))
            screen.blit(text, (200, 100))

            font = pygame.font.Font(None, 30)
            text = font.render("Enter Your Name:", True, (255, 255, 255))
            screen.blit(text, (200, 200))

            pygame.draw.rect(screen, (255, 255, 255), (300, 250, 200, 50))
            font = pygame.font.Font(None, 30)
            text = font.render(player_name, True, (0, 0, 0))
            screen.blit(text, (310, 260))

            pygame.draw.rect(screen, (0, 255, 0), (300, 300, 200, 50))
            font = pygame.font.Font(None, 30)
            text = font.render("Let's Draw!", True, (0, 0, 0))
            screen.blit(text, (350, 310))
            
            pygame.draw.rect(screen, (0, 0, 255), (650, 10, 140, 40))
            font = pygame.font.Font(None, 25)
            text = font.render("Toggle Fullscreen", True, (255, 255, 255))
            screen.blit(text, (655, 20))

        elif end_screen:
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 50)
            text = font.render("Time's Up!", True, (255, 255, 255))
            screen.blit(text, (300, 100))

            pygame.draw.rect(screen, (0, 255, 0), retry_button)
            font = pygame.font.Font(None, 30)
            text = font.render("Retry", True, (0, 0, 0))
            screen.blit(text, (380, 310))

        else:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally to correct for mirrored video feed
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB format for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks using Mediapipe
            results = hands.process(image)

            # Create a black mask to blackout the webcam feed
            mask = np.zeros_like(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections on the original frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Draw landmarks on the mask in white (255) to reveal hands
                    mp_drawing.draw_landmarks(mask, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

                    # Get the index finger tip and thumb tip positions
                    index_finger_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                                        int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))
                    thumb_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
                                 int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]))

                    # Check if the distance between thumb tip and index finger tip is small enough to indicate touch
                    touch_distance_threshold = 50
                    if np.linalg.norm(np.array(index_finger_tip) - np.array(thumb_tip)) < touch_distance_threshold:
                        drawing = True
                        # Add the index finger tip to the drawing_points list
                        drawing_points.append(index_finger_tip)
                    else:
                        drawing = False

                    if pencil_mode and drawing:
                        # Draw lines when in pencil mode and actively drawing
                        if len(drawing_points) >= 2:
                            for i in range(1, len(drawing_points)):
                                cv2.line(frame, drawing_points[i - 1], drawing_points[i], (0, 255, 0), 5)
                                pygame.draw.line(pencil_surface, (0, 255, 0), drawing_points[i - 1], drawing_points[i], 5)

            # Combine the original frame and the mask to blackout everything except hands
            blackout_frame = cv2.bitwise_and(frame, mask)

            # Flip the frame vertically (optional, adjust as needed)
            blackout_frame = cv2.flip(blackout_frame, 1)

            # Rotate the frame by the current angle
            rotated_frame = rotate_frame(blackout_frame, angle)

            # Convert the frame back to BGR for displaying with Pygame
            rotated_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_RGB2BGR)

            # Convert the frame to Pygame surface
            frame_surface = pygame.surfarray.make_surface(rotated_frame)

            # Render the hand detection feed in Pygame window, centered in the window
            screen.blit(frame_surface, (x_offset, y_offset))

            # Render the pencil surface on top of the webcam feed
            screen.blit(pencil_surface, (x_offset, y_offset))

            if game_timer_start:
                time_elapsed = datetime.now() - game_timer_start
                time_left = max(game_duration - time_elapsed, timedelta(seconds=0))
                draw_timer(screen, time_left.seconds, window_width)

                if time_left.total_seconds() <= 0:
                    # Save the pencil surface as an image file
                    image_folder = 'static/images'
                    image_name = f"{player_name}_drawing.png"
                    image_path = os.path.join(image_folder, image_name)
                    pygame.image.save(pencil_surface, image_path)
                    end_screen = True

        pygame.display.update()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
