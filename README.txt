Easy to Use Trading Pricing Tool


USE THIS AT YOUR OWN RISK
It is possible that this will set off scanners that will ban your account.
This is mitigated by using only screenshots and no input automation.

This is a tool that allows you to instantly determine the value of a trade given the parts.
Set your preferred values for the parts in the GUI.
Click activate and the program will automatically tell you the value of the trade.
This MASSIVELY speeds up trades and reduces mental strain.

Startup takes a while

ctrl resets the pricing tool

If you run the price extractor file, it will pull ducat values from the WF wiki.
Know that some of the values on the wiki are incorrect. Use warframe.market to know the correct values.
Known incorrect values and their correct values:
Limbo Prime Neuroptics: 100
Rhino Prime Chassis: 65
Gauss Prime Blueprint: 25

Ducat values can be editted using MS Excel, Pycharm, or any other program that can edit CSV files.







Ask chatgpt for instructions if needed

Instructions:
Install Tesseract on PC to your C drive
The tool looks for 'C:\Program Files\Tesseract-OCR\tesseract.exe'

Add tesseract to path during install or add it manually later.

Install CUDA Toolkit 12.3. and cuDNN library. Use the latest version of these if a new version is out.
This is optional but improves performance massively. 
It takes load off CPU and uses GPU instead, which is much faster.
or don't do this to save about 500mb of RAM at the cost of big CPU usage.



To create the .exe...

Open pycharm or your preferred IDE
Open this folder as a project
Create a venv for this if necessary
Open main.py
Open terminal
Run:
.\venv\Scripts\activate
Run:
pip install -r requirements.txt
pip install the appropriate and latest version of pytorch manually 
Test the program by running the main.py file in pycharm
Run:
pyinstaller --onefile --name="Prime Junk Counter" main.py
It will take a while
The .exe will be in the dist folder
Move the .exe to the main project folder (unknown if necessary)
Use shortcut to access .exe (preferred but possibly not necessary)
or press windows key and search for prime junk counter after running .exe








