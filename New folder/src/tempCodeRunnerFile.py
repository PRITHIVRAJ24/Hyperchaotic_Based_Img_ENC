def dna_mapping(image, key=54321):
    np.random.seed(key)
    mask = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    return cv2.bitwise_xor(image, mask)