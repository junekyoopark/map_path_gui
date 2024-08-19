# Clothoid Path Generation GUI (with satellite imagery background)

### Usage
Note: if you run python using `python` instead of `python3`, then `subprocess.run(['python3', 'clothoid-fitting/z_calc.py']` and all other related lines should be changed to something like `subprocess.run(['python', 'clothoid-fitting/z_calc.py']`
1. Download and install Gurobi from their website. Then activate your license.
2. Install required libraries listed in `requirements.txt`.
3. Create a directory above this one called `config` and create a `naver_api.txt` file inside with the following syntax:
    ```
    CLIENT_ID = 'id'
    CLIENT_SECRET = 'secret'
    ```
    Replace `id` and `secret` with yours

4. Run `map_gui_uvland.py` or `map_gui_robotland` or `map_gui_kunsan.py`.
5. ??? (it's a GUI, so... click buttons I guess?)
6. Profit
The `SMOOTH-Z` csv values are saved in `output/clothoidal_path_3d.csv`.

### Why?
For use in Yonsei Drone (https://yonseidrone.com) and MECar.

### Sources
The `clothoid-fitting` repository was used: https://github.com/junioranderson12/clothoid-fitting 