
"""
#########################
Utils (``utils.utils``)
#########################
"""

class PTFInputError(Exception):
    """
    Generic Python exception derived object raised by patefy library. 
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class PTFConvergenceError(Exception):
    """
    Generic Python exception derived object raised by patefy library. 
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
