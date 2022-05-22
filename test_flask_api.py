try:
    from flask_api import app
    import unittest
except Exception as ex:
    print(f"Exception while importing unittest modules  with error {ex}")


class Flask_API_Test(unittest.TestCase):

    #Check if response 200
    def test_response(self):
        tester = app.test_client(self)
        response = tester.get("/")
        statuscode = response.status_code
        self.assertEqual(statuscode,200)

    #Check if response is json
    def test_response_json(self):
        tester = app.test_client(self)
        response = tester.get("/")
        self.assertEqual(response.content_type, "application/json")

    #Check for first empty run
    def test_data_first_run(self):
        tester = app.test_client(self)
        response = tester.get("/")
        self.assertTrue(b"None" in response.data)

if __name__=="__main__":
    unittest.main()