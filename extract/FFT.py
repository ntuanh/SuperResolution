# extract raw image to frequency chart
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    # 3. get high-frequency
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    radius = 30
    high_freq = magnitude.copy()
    high_freq[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0

    # 4. compute complexity
    complexity = np.mean(high_freq)

    # 5. display text
    cv2.putText(frame, f'Complexity: {int(complexity)}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # 6. display FFT spectrum (optional)
    spectrum = 20 * np.log(magnitude + 1)
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum = spectrum.astype(np.uint8)

    cv2.imshow("Webcam", frame)
    cv2.imshow("FFT Spectrum", spectrum)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()