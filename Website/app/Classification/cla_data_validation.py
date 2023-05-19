"""
Author: Vishal Kundar and Vedant Barbhaya

data_validation.py takes the data file given by user and performs validation
checks. The steps include:

    1. Identifying type of data file.(csv xlsx tsv json)

    2. Converting to data frame
    
    3. Validation check of data (2 columns min. 100000 > Rows > 300 allowed 
       data types: num, text) (Clean, format and quality check).No two columns
       headers should be same.

    4. Function to display data to user to select dependant variable

    5. Identifying if problem is classification or regression based on the
    dependant variable

"""

#Packages
import pandas as pd
import numpy as np

class data_check:
    """
    Consists of methods to deal with the above steps.

    Class variables
    ---------------
    __filename: TYPE: Private class string variable.
            DESCRIPTION: Holds filename given by user
    __filetype: TYPE: Private class string variable.
            DESCRIPTION: Holds filetype found using identify_file() method
    __fileValid: TYPE: Private class boolean variable.
            DESCRIPTION: If the file is found to have problems it prevents
            usage of file_to_dataframe() and identify_problem() methods

    """

    def __init__(self, filename):
        """
        Constructor to initialize class. Initializes class variable filename
        with given file name and filetype to empty string.

        Parameters
        ----------
        filename : TYPE : String
            DESCRIPTION : Name of file entered by user.

        Returns
        -------
        None

        """
        #private class variables
        self.__filename = filename
        self.__filetype = ""
        self.__fileValid = False


    def identify_file(self):
        """
        identifying type of data file.(csv xlsx tsv json)

        Sets filetype and filevalid variable based on whether or not file is
        valid.

        Parameters
        ----------
        None

        Returns
        -------
        None if file is of acceptable else returns log message that file type
        cannot be used.

        """
        self.__filetype = ""
        self.__fileValid = False

        allowed_ext = ["csv","tsv","xlsx","json"]
        extension = self.__filename.split(".")[1]

        if extension in allowed_ext:
            self.__filetype = extension
            self.__fileValid = True
            return "None"

        else:
            return "File type not supported!"

    def validation_check(self, df):
        """
        Validation check of data (2 columns min. 100000 > Rows > 300 allowed
        data types: num, text) (Clean, format and quality check). No two columns
        headers should be same.

        Sets fileValid variable based on whether or not it is valid.

        Parameters
        ----------
        df : TYPE : Dataframe
             DESCRIPTION : Dataframe consisting of data extracted from file.

        Returns
        -------
        None if file is of acceptable type else returns log message that file
        has issues.

        """
        if self.__fileValid == False:
            return "Error: File type not valid! Accepted - [csv, tsv, json, xlsx]"

        else:
            self.__fileValid = False
            self.__no_cols = len(df.columns)
            self.__no_rows = df.shape[0]

            if self.__no_cols < 2:
                return "Error: Insufficent number of features!"

            if self.__no_rows < 1 or self.__no_rows > 100000: #300
                return "Error: Number of examples out of range! 100000 > examples > 300"

            self.__fileValid = True
            return "None"

    def file_to_dataframe(self):
        """
        Converting file type to pandas data frame

        Parameters
        ----------
        None

        Returns
        -------
         dataframe consisting of file data.

        """
        if self.__fileValid == False:
            return "File not valid!"
        else:
            if (self.__filetype == "csv"):
                return pd.read_csv(self.__filename)

            elif (self.__filetype == "tsv"):
                return pd.read_csv(self.__filename, sep='\t')

            elif(self.__filetype == "xlsx"):
                #expects single sheet only in xlsx file
                return pd.read_excel(self.__filename, sheet_name=None)

            elif (self.__filetype == "json"):
                #try this out
                return pd.read_json(self.__filename)
            else:
                return """File type not supported! File type supported -> csv,
            tsv, json and xlsx. File type given -> """ + self.__filetype

    def identify_problem(self, df, y):
        """
        Identifying if problem is classification or regression based on the
        dependant variable

        Parameters
        ----------
        df : TYPE : Dataframe
             DESCRIPTION : Dataframe consisting of data extracted from file.

        y : TYPE : String
             DESCRIPTION : Column header of dependant variable

        Returns
        -------
         String stating if the problem is regression or classification. If any
         problem occurs then log message is sent back.

        """
        if self.__fileValid == False:
            return "Error: File/Data not valid!"
        else:
            try:
                data = df[y].values

                if(len(np.unique(data)) > 10):
                    return "Regression"
                else:
                    return "Classification"

            except KeyError:
                return "Error: column header not found!"

            except Exception as e:
                return "Error: " + str(e)
