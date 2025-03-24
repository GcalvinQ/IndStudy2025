# IndStudy2025
Independent study expanding and elevating research done by myself and a classmate previously

**The Original**
Took research findings regarding the fully homomorphic GGH encryption method prototype; lacked full Quantum Safe encryption, and was not texted using LLL attack methods 

**The update**
Uses LWE encryption method and compresses the encryption image to allow for LLL attack security, and still fully homomorphic, but with additional Gaussian noise, lattice dimensions for 256 encryption ability,and prime a modulus for lattice creation. 



**Requirements for Rough Draft**
pip install pillow
pip install numpy
pip install sympy
pip install tqdm

**Issues with Initial Rough Draft**:
I was unable to use a noise factor using Gausian Noise (constant value sigma) when decrypting, this will be a push forward for the final draft. Also am having issues preserving the input size as well as having to use a known lattice that is 100% invertable, ran into runtime issues when creating a complete random lattice for the baasis to be constructed from. 

