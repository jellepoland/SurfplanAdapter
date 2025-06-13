# Test the current profile indexing logic
n_ribs = 18
k = n_ribs // 2  # k = 9

print('Current indexing logic:')
print('n_ribs =', n_ribs, ', k =', k)
print('Rib index -> Profile name:')

for i in range(n_ribs):
    # Current logic
    if n_ribs % 2 == 1:
        profile_name = f'prof_{1 + abs(-k + i)}'
    else:
        if i < k:
            profile_name = f'prof_{k - i}'
        else:
            profile_name = f'prof_{i - k + 1}'
    print(f'  rib[{i}] -> {profile_name}')

print('\n' + '='*50)
print('What should it be for strict left-to-right mapping:')
print('If we want ribs to map directly to profile numbers:')

for i in range(n_ribs):
    profile_num = i + 1  # Direct mapping: rib[0] -> prof_1, rib[1] -> prof_2, etc.
    print(f'  rib[{i}] -> prof_{profile_num}')
