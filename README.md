# Image-Processing
# Assignment 1

**1) Jai Hind: Drone Positioner**

**Objective:** Create a reference image for a drone to adjust its position based on an initial input image. The drone sees the national flag and needs to adjust its position until it matches a reference image.

**Approach:** The program should process the input image, adjust it to the desired reference parameters (600x600 pixels, specific circle dimensions), and return the corrected image.

**2) Eent Ka Jawab Image Se: Brick Quality**

**Objective:** Analyze brick quality by converting audio files of bricks being struck into spectrograms and classify them as either 'metal' (high quality) or 'cardboard' (low quality).

**Approach:** Use Python libraries like librosa to convert audio to spectrograms, then design an algorithm to classify the quality based on the resulting images.

**3) Anuprasth Drishyam: Horizontal Viewer**

**Objective:** Correct the alignment of scanned images containing Sanskrit text so they are horizontally aligned for optimal OCR processing.

**Approach:** The program should detect the orientation of text in the input images and rotate them to be perfectly horizontal, ensuring no characters are cut off in the process.

# Assignment 2

**1) The Lava: Antaragni**

**Objective:** Detect and overlay lava flow regions on given images, producing a binary mask where lava regions are white, and non-lava regions are black.

**Approach:** Implement image segmentation techniques to accurately identify and highlight lava flow areas on drone-captured images.

**2) Pro-night with or without Camera Flash?: Prolight**

**Objective:** Fuse images taken with and without a flash to create a final image that balances the benefits of both, reducing flashy artifacts while preserving image quality.

**Approach:** Utilize a cross bilateral filter or similar method to combine the two images, focusing on minimizing noise while enhancing the overall appearance.

**3) The Victory Over Delusion: Dussehra**

**Objective:** Determine whether an image of Ravana is real or manipulated (fake), i.e., if the demon king is genuinely facing the viewer or if his image is deceitfully altered.

**Approach:** Design a program that analyzes the image for signs of manipulation and classifies it as 'real' or 'fake' based on predefined criteria.
