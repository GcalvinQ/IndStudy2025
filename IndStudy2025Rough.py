from PIL import Image
import numpy as np
import sympy as sp
from tqdm import tqdm
import logging

# logging for debugging
# I was having runtime issues, allowed me to see what section was getting bogged down
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def modinv(a, q):
    ###
#     Compute the modular inverse of a modulo q using the Extended Euclidean Algorithm.
#     This does not rely on negative exponents.
    ###
    a = int(a)
    t, newt = 0, 1
    r, newr = q, a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError(f"{a} is not invertible modulo {q}")
    if t < 0:
        t += q
    return t

def invert_lower_triangular(L, q):
    # Compute the modular inverse of a lower-triangular matrix L modulo q.
    n = L.shape[0]
    L_inv = np.zeros_like(L)
    for i in range(n):
        L_inv[i, i] = modinv(L[i, i], q)
        for j in range(i):
            s = 0
            for k in range(j, i):
                s = (s + L[i, k] * L_inv[k, j]) % q
            L_inv[i, j] = (-modinv(L[i, i], q) * s) % q
    return L_inv

def generate_invertible_matrix(n, q):
    ##
#     invertible n x n matrix A over Z_q by:
#     1. Creating a random lower-triangular matrix L with nonzero diagonal entries.
#     2. Creating a random permutation matrix P.
#     3. Computing A = P · L mod q.
#     A, L, and P.
    ##
    # Generate random lower-triangular matrix L with entries in [0, q)
    L = np.tril(np.random.randint(0, q, size=(n, n)))
    # Ensure nonzero diagonal entries (choose values in 1...q-1)
    for i in range(n):
        if L[i, i] == 0:
            L[i, i] = np.random.randint(1, q)
    
    # Generate a random permutation matrix P
    P = np.eye(n)[np.random.permutation(n)]
    
    # Compute A = P · L mod q
    A = (P.dot(L)) % q
    return A, L, P

class LWEEncryption:
    def __init__(self, n, q, sigma):
        self.n = n            # Lattice/block dimension
        self.q = q            # Prime modulus, this value is extremely high for reasons stated in the LWE paper that is within the git
        self.sigma = sigma    # Standard deviation for Gaussian noise

        logging.info("Generating structured invertible matrix A...")
        self.A, self.L, self.P = generate_invertible_matrix(self.n, self.q)
        
        logging.info("Computing modular inverse of L...")
        L_inv = invert_lower_triangular(self.L, self.q)
        # Since A = P · L, then A_inv = L_inv · Pᵀ (because P is orthogonal: P⁻¹ = Pᵀ)
        self.A_inv = (L_inv.dot(self.P.T)) % self.q
        logging.info("Key generation complete.")

    def encrypt(self, message):
        message_vector = message.flatten()
        original_length = len(message_vector)
        
        remainder = original_length % self.n
        padding_size = self.n - remainder if remainder != 0 else 0
        message_vector = np.pad(message_vector, (0, padding_size), 'constant')
        message_blocks = message_vector.reshape(-1, self.n)
        
        noise = np.random.normal(0, self.sigma, size=message_blocks.shape).astype(int) % self.q
        encrypted_blocks = (np.dot(message_blocks, self.A) + noise) % self.q
        encrypted = encrypted_blocks.flatten()[:original_length]
        return encrypted

    def decrypt(self, ciphertext, original_shape):
        total_elements = np.prod(original_shape)
        remainder = ciphertext.size % self.n
        if remainder != 0:
            ciphertext = np.pad(ciphertext, (0, self.n - remainder), 'constant')
        ciphertext_blocks = ciphertext.reshape(-1, self.n)
        decrypted_blocks = (np.dot(ciphertext_blocks, self.A_inv)) % self.q
        decrypted = decrypted_blocks.flatten()[:total_elements]
        decrypted = decrypted.astype(int) % 256
        decrypted = decrypted.reshape(original_shape)
        return decrypted

class ImageProcessor:
    def __init__(self, n, q, sigma):
        self.lwe = LWEEncryption(n, q, sigma)

    def load_image_to_array(self, file_path, target_size=None):
        img = Image.open(file_path)
        img = img.convert('RGB')
        if target_size:
            img = img.resize(target_size)
        img_array = np.array(img)
        return img_array

    def compress_image(self, file_path):
        img = Image.open(file_path)
        compressed_path = file_path.replace('.jpg', '_compressed.jpg')
        img.save(compressed_path, format="JPEG", quality=50)
        return compressed_path

    def export_image(self, image_array, file_path):
        image_array = np.clip(image_array, 0, 255).astype('uint8')
        img = Image.fromarray(image_array)
        img.save(file_path)
        img.show() # Showing both images for debugging
        logging.info(f"Image saved as {file_path}")

    def encrypt_image(self, image_array):
        h, w, c = image_array.shape
        encrypted_channels = []
        logging.info("Starting encryption of image channels...")
        for i in tqdm(range(c), desc="Encrypting channels"):
            encrypted = self.lwe.encrypt(image_array[:, :, i])
            total_pixels = h * w
            if encrypted.size > total_pixels:
                encrypted = encrypted[:total_pixels]
            elif encrypted.size < total_pixels:
                encrypted = np.pad(encrypted, (0, total_pixels - encrypted.size), 'constant')
            encrypted_channels.append(encrypted.reshape(h, w))
        encrypted_image = np.stack(encrypted_channels, axis=-1)
        logging.info("Encryption complete.")
        return encrypted_image

    def decrypt_image(self, encrypted_image, original_shape):
        h, w, c = original_shape
        decrypted_channels = []
        logging.info("Starting decryption of image channels...")
        for i in tqdm(range(c), desc="Decrypting channels"):
            channel_encrypted = encrypted_image[:, :, i].flatten()
            decrypted = self.lwe.decrypt(channel_encrypted, (h, w))
            decrypted_channels.append(decrypted)
        decrypted_image = np.stack(decrypted_channels, axis=-1)
        logging.info("Decryption complete.")
        return decrypted_image

# MAIN/Constants
n = 256    # Lattice/block dimension
q = 3329   # Prime modulus (must be > 255)
# do not understadn why with any noise the image cannot be relayed after decrytion 
sigma = 0  # Standard deviation for Gaussian noise

# Initialize ImageProcessor with LWE-based encryption
image_processor = ImageProcessor(n, q, sigma)

# === IMAGE PROCESSING ===
input_file = "Cat_ImageJPG.jpg"  # Ensure this file is in your working directory
logging.info("Compressing image...")
compressed_file = image_processor.compress_image(input_file)
logging.info("Loading image into array...")
image_array = image_processor.load_image_to_array(compressed_file, target_size=(256, 256))

logging.info("Encrypting image...")
encrypted_image = image_processor.encrypt_image(image_array)
image_processor.export_image(encrypted_image, "encrypted_image.jpg")
logging.info(f"Encrypted image shape: {encrypted_image.shape}")

logging.info("Decrypting image...")
decrypted_image = image_processor.decrypt_image(encrypted_image, image_array.shape)

output_file = "decrypted_image.jpg"
image_processor.export_image(decrypted_image, output_file)
