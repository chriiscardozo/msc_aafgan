import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig, ax = plt.subplots()
years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
qty = [1, 3, 34, 293, 741, 1097, 1221, 777]

for i in range(len(qty)):
    plt.annotate(str(qty[i]), xy=(years[i],qty[i]), ha='center', va='bottom')

ax.set_ylabel('Quantity')
ax.set_xlabel('Year')

ax.bar(years,qty)

plt.title('Number of publications related to GANs (dblp indexed search)')


plt.show()