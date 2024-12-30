USE THIS AT YOUR OWN RISK

Ask chatgpt for instructions if needed

Install Tesseract on PC to your C drive
The counter looks for 'C:\Program Files\Tesseract-OCR\tesseract.exe'

Add tesseract to path

install pytorch

install CUDA Toolkit 12.3. and cuDNN library or whatever is up to date.
to improve performance massively. 
takes load off CPU and uses GPU efficiently instead.
or don't to save about 500mb of RAM at the cost of big CPU usage

To create the .exe,
open pycharm
open this folder as a project
create a venv for this if necessary
open main.py
open terminal
run:
.\venv\Scripts\activate
run:
pip install -r requirements.txt
test it by running the main.py file in pycharm
run:
pyinstaller --icon=ducat_icon.png --onefile --name="Prime Junk Counter" main.py
it will take a while
the .exe will be in the dist folder
move the .exe to the main project folder (unknown if necessary)
use shortcut to access .exe (preferred but possibly not necessary)
or press windows key and search for prime junk counter after running .exe


ctrl is reset counter

If it is lagging, don't click multiple times because it will remember that... maybe


Startup takes a while

Slot position settings might not save so remember what you put in. Try some testing to see what happens. Idk what happens.





If you run the price extractor file, it will pull ducat values from the WF wiki.
Know that some of the values are incorrect.
Ducat values can be editted using MS Excel, Pycharm, or any other program that can edit CSV files.

List of actual values for known incorrect values:
Rhino Prime Chassis: 65




