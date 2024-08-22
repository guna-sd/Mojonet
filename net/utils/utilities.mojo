fn handle_issue(msg: String):
    print("Issue: " + msg)
    exit(1)


# struct Status:
#     """
#     The `Status` struct represents a status result that can be either a boolean or a string.
#     """

#     alias type = Variant[Bool, String]
#     var result: Self.type
#     """The value of the status, which can be either a boolean or a string."""

#     fn __init__(inout self: Self, arg: Bool):
#         self.result = Variant[Bool, String](arg)

#     fn __init__(inout self: Self, arg: String):
#         self.result = Variant[Bool, String](arg)

#     fn __str__(self: Self) -> String:
#         if self.result.isa[Bool]():
#             return self.result.__getitem__[Bool]().__str__()
#         elif self.result.isa[String]():
#             return self.result.__getitem__[String]()
#         else:
#             return "Invalid type"

#     fn is_bool(self: Self) -> Bool:
#         """
#         Checks if the status result is a boolean.
#         """
#         return self.result.isa[Bool]()

#     fn is_string(self: Self) -> Bool:
#         """
#         Checks if the status result is a string.
#         """
#         return self.result.isa[String]()
