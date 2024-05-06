This assignment was primarily tested on a Jupyter Notebook and Visual Studio Code on my local machine, rather than on Google Colab.

According to Plotly documentation[https://plotly.com/python/webgl-vs-svg/]: It may not be possible to render more than 8 WebGL-involving figures on the same page at the same time. Since Plotly cannot display many figures in a single IPYNB file, we have divided our file into several parts to facilitate a more comprehensive review of all results.

You may directly view the "*_ALL.ipynb" notebook on Jupyter Notebook or the "*_ALL_COLAB.ipynb" on Google Colab. Note that as you continue viewing, some earlier plots may disappear. Here are four options to deal with this:
1. (Most stable method) Open "_ALL.ipynb" in Google Colab or Jupyter Notebook and execute code blocks one by one to ensure that the most recently executed block will display the plot. Alternatively, you can apply the same method to the "part*.ipynb" files.
2. View "*_ALL.ipynb" in Visual Studio Code (tested on Windows 11) with the file set to "Trusted."
3. Press the "Download as png" button in each Plotly interactive panel on the top right to save a static image of the plots.
4. View the "part*.ipynb" files in Jupyter Notebook, which are divided into sections to ensure no plots disappear.
5. Refresh the notebook page in your browser, then locate and interact with your desired plot.

All discussions are based on the outputs from my notebook execution. Since the PyTorch3D optimization involves randomness, there might be significant differences if you run it again compared to my notebook. Additionally, some discussions include strikethrough text to explain phenomena observed in outputs from different executions.

The "*_ALL.ipynb" notebook contains all the source code for my assignment, and the "part*" files function in the same way as this notebook. If you want to experiment with the "*_ALL.ipynb" notebook, you can either "Restart the kernel and run all cells" or execute it block by block. However, avoid running any previous blocks more than once as it could cause unexpected behavior.