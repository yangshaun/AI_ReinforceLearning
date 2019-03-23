import collections
import numpy as np
import time
class mdp:
	utility = list()
	car_reward = list()
	array_size = 0
	states_squence = set()
	start_point = []
	end_point = []
	obstacle = set()
	cars = 0
	all_score = []

	def preprocessing(self):
		with open('./input.txt','r') as file:
			line = [l.rstrip() for l in file]

		line = collections.deque(line)
		self.array_size = int(line.popleft())
		self.cars = int(line.popleft())
		num_obstacle = int(line.popleft())

		
		for i in range(num_obstacle):
			current = str(line.popleft()).split(',')
			self.obstacle.add(tuple([int(x) for x in current]))		

		for i in range(self.cars):
			current = str(line.popleft()).split(',')
			self.start_point.append([int(x) for x in current])		
		for i in range(self.cars):
			current = str(line.popleft()).split(',')
			self.end_point.append(tuple([int(x) for x in current]))

		self.car_reward = [np.full((self.array_size ,self.array_size ),-1,float) for i in range(self.cars)]

		# for i in range(len(self.start_point)):
		# 	self.car_reward[i][self.start_point[i][0]][self.start_point[i][1]] = 0

		for i in range(len(self.end_point)):
			self.car_reward[i][self.end_point[i][0]][self.end_point[i][1]] += 100
			for obs in self.obstacle:
				self.car_reward[i][obs[0]][obs[1]] -= 100

		for i in range(self.array_size):
			for j in range(self.array_size):
				self.states_squence.add((i,j))

	def turn_right(self,direct):
		if direct =='N':
			return 'E'
		elif direct =='W':
			return 'N'
		elif direct == 'E':
			return 'S'
		else: 
			return 'W'
	def turn_left(self,direct):
		if direct =='N':
			return 'W'
		elif direct =='W':
			return 'S'
		elif direct == 'E':
			return 'N'
		else: 
			return 'E'
	def go (self,state,direct):
		if direct =='N':
			if state[0] - 1 < 0:
				return state
			else:
				return (state[0] - 1,state[1])			

		elif direct =='E':
			if state[1] + 1 >= self.array_size:
				return state
			else:
				return (state[0] , state[1] + 1)

		elif direct == 'W':
			if state[1] - 1 < 0:
				return state
			else:
				return (state[0],state[1] - 1)
		else:
			if state[0] + 1 >= self.array_size:
				return state
			else:
				return (state[0] + 1 , state[1])
	def Trans(self, state, action):
		if action == None:
			return [(0.0, state)]
		else:
			return [(0.7, self.go(state, action)),
					(0.1, self.go(state, self.turn_right(action))),
					(0.1, self.go(state, self.turn_left(action))),
					(0.1, self.go(state, self.turn_left(self.turn_left(action))))]

	def value_iteration(self,U1,final_direct_table,car_index, ratio = 0.1, r = 0.9):
		
		current_reward  = self.car_reward[car_index]
		order = {"N":4,"S":3,"E":2,"W":1}	
		while True:
			U = U1.copy()
			delta = 0.0
			for i in range(self.array_size):
				for j in range(self.array_size):
					s = (i,j)
					cell_max = float('-inf')
					if s[0] == self.end_point[car_index][0] and s[1] == self.end_point[car_index][1]:
						final_direct_table[s[0]][s[1]] = str(current_reward[(s[0],s[1])])
						cell_max = max(cell_max,0.0)
					else:	
						for a in ['N','W','E','S']:
							summation = sum([p * U[s1] for (p, s1) in self.Trans(s, a)])
							if summation > cell_max:
								cell_max = summation
								final_direct_table[s[0]][s[1]] = str(a)
							if summation == cell_max:
								if order[final_direct_table[s[0]][s[1]]] < order[a]:		
									final_direct_table[s[0]][s[1]] = str(a)

					U1[s] = current_reward[s[0]][s[1]] + r * cell_max
					delta = max(delta, abs(U1[s] - U[s]))

			if delta < ratio:			
				return final_direct_table

	def run_policy(self,final_direct_table):
		all_cars_score = []
		for i in range(self.cars):
			reward_table = self.car_reward[i]
			current_direct_table = final_direct_table[i]
			current_car_score = 0.0

			for j in range(10):
				pos = self.start_point[i]
				np.random.seed(j)
				swerve = np.random.random_sample(1000000)
				k = 0
				while pos[0] != self.end_point[i][0] or pos[1] != self.end_point[i][1]:	
					move = current_direct_table[pos[0]][pos[1]]					
					if swerve[k] > 0.7:
						if swerve[k] > 0.8:
							if swerve[k] > 0.9:
								move = self.turn_left(self.turn_left(move))
							else:
								move = self.turn_left(move)													
						else:
							move = self.turn_right(move)
					pos = self.go(pos,move)
					current_car_score += reward_table[pos[0]][pos[1]]
					k += 1					
			all_cars_score.append(np.floor(current_car_score / 10.0))
		return all_cars_score

	def main(self):
		
		self.preprocessing()
		cars_direct_table = []

		for i in range(self.cars):
			U2 = dict([(s,0.0) for s in self.states_squence])
			f2 = [[ "" for x in range(self.array_size)]for j in range(self.array_size)]
			cars_direct_table.append(self.value_iteration(U2,f2,i))	

		return self.run_policy(cars_direct_table)
		

if __name__ == '__main__':	
	a = time.time()	
	with open('./output.txt','w') as out:
		for result in mdp().main():
			out.write('%d\n'%result)
	b = time.time()
	print b - a
	
