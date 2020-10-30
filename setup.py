# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:02:43 2020

@author: Abhilash
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:45:49 2020

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'MiniClassifier',         
  packages = ['MiniClassifier'],   
  version = '0.1',       
  license='MIT',        
  description = 'A Classifier Deep Learning library for Binary/Categorical Text Classification with Keras',   
  long_description='This library contains a deep learning architecture comprised of Bidirectional LSTMs, Dense and Convolution Neural Networks for Text Classification(Keras).There are parameters for using Pre-trained embeddings like Glove, Fast text etc.For using pretrianed embeddings,a relative path to the embedding file should be provided.The entire workflow is provided in Test Scripts,for Binary and Categorical Classification.Only changes are required in the relative paths of the dataset, embeddings(if any) and the hyperparameters.',
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/MiniClassifier',   
  download_url = 'https://github.com/abhilash1910/MiniClassifier/archive/v_01.tar.gz',    
  keywords = ['Text Classification','Embeddings','Deep Learning Networks','Bidirectional Lstms','Convolution Networks','Pretrained Embeddings','Keras','Tensorflow'],   
  install_requires=[           

          'numpy',         
          'keras',
          'tensorflow',
          'pandas',
          'sklearn',
          'nltk',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
