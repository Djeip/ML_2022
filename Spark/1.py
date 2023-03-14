class HUB:
    pass


class Team(HUB):
    def __init__(self, team, owner, products,min_workers):
        self.team = team
        self.owner = owner
        self.products = products
        self.workers_min = min_workers
        self.workers = []

    def workflow(self):
        pass


class PotentialWorker:
    def __init__(self):

    def coffee(self):
        pass

    def toxic(self):
        pass

    def work(self):


    def business_planning(self):
        while True:
