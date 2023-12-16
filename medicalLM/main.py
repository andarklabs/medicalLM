""" Run file for our model code.

* The purpose of this file is to create an easy to use helper file that
* interacts with all of our classes needed for the model.

"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
