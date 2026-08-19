from setuptools import setup, find_packages

exec(open('eat/version.py').read())

setup(
  name = 'exposure-aware-training',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = 'Exposure-Aware Training (EAT): Lightweight degradation-based training strategy for low-light object detection without target-domain data',
  author = 'Yawen Su',
  author_email = 'syw123456_0617@qq.com',
  url = 'https://github.com/sss84/EAT',
  long_description_content_type = 'text/markdown',
  keywords = [
    'exposure-aware-training',
    'low-light-object-detection',
    'domain-generalization',
    'degradation-modeling',
    'zero-inference-adaptation'
  ],
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch>=0.4.2',
    'numpy',
    'pillow',
    'pytorch-fid',
    'scipy',
    'torch>=2.0',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
