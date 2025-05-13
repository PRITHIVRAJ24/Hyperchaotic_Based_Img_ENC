# Hyperchaotic-Based Image Encryption

This project implements a secure and efficient image encryption scheme using a hyperchaotic system. The scheme combines multiple rounds of confusion and diffusion processes with DNA-based operations to enhance security and protect image data against cryptographic attacks.

## 🔒 Features

- **Hyperchaotic System**: Utilizes a Chen–Lorenz hyperchaotic system for generating pseudo-random sequences.
- **DNA Encoding**: Employs DNA encoding and XOR-based operations to obscure pixel values.
- **Multi-round Encryption**: Applies multiple rounds of confusion and diffusion for enhanced resistance to statistical and differential attacks.
- **Color Image Support**: Works on RGB color images, encrypting each channel independently.
- **Channel Shuffling**: Randomly rearranges color channels to add additional confusion.
- **No Bit Scrambling**: Ensures faster encryption by avoiding complex bit-level operations.

## 🧠 Encryption Workflow

1. **Key Generation** using hyperchaotic initial conditions.
2. **Image Preprocessing** including RGB channel separation and resizing.
3. **DNA Encoding** of pixel values.
4. **XOR Operation** between DNA sequences and chaotic key streams.
5. **Confusion & Diffusion** using chaotic permutation and dynamic diffusion.
6. **Final Assembly** of the encrypted image.

## 🚀 Getting Started

### Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib (for visualization)

### Installation

```bash
pip install -r requirements.txt
