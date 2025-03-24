from setuptools import setup, find_packages

setup(
    name='Quantum_Fokker_Planck-main',  # Replace with your project name
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',          # For numerical computations
        'scipy',          # For advanced numerical computations
        'matplotlib',     # For plotting
        'tqdm',           # For progress bars
        'sympy',          # For symbolic mathematics
        'imageio[ffmpeg]',  # For reading and writing video files with ffmpeg support
    ],
    
    author='Wen-Hua Wu',  
    author_email='aw106@rice.edu',
    description='A code simulating the time evolution under quantum fokker-planck equation', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',  # Use your project's license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Optional: Specify the Python version
)