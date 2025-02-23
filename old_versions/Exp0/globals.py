
import neptune.new as neptune

def initialize():

    # Initialize Neptune and create new Neptune Run
    global run
    run = neptune.init_run(
        project="itohamy/NCP-flownet",
        tags="NCP_EB_new",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZWU1MWE2ZS0yZjEwLTQzMTItODdlYi1kN2I4ODgzMDA4M2IifQ==",
        source_files=["*.py"]
    )