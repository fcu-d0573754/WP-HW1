# ADL HW2

SIYUE HU 
r08922159


## Train
1. bash ./download.sh
2. unzip data.zip
3. create model folder
```
mkdir -p model 
```
4. Run training script
```
python run_squad.py --do_train
```

## Test
1. bash ./download.sh
2. bash run.sh --input_file $1 --output_path $2

## Plot figures
### Cumulative Answer Length
```
def draw_hist(data):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot the cumulative histogram
    n, bins, patches = ax.hist(data, 30, density=True, histtype='step',
                           cumulative=True)
    ax.plot(kind='kde')    
    plt.xlabel("Length")
    plt.ylabel("Count(%)")
    plt.title("Cumulative Answer Length")

    show_plot_visdom()
    plt.show()
    plt.close()


def evaluate_and_draw_hist():
    examples = process.read_examples(input_file='./data/train.json', is_train= True)
    save = []
    for example in examples:
        if len(example.answers_text) > 0:
            context_token = tokenizer.tokenize(example.answers_text)
            save.append(len(context_token))
    
    data = np.array(save)
    draw_hist(data)
```

### Answerable Threshold
```
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Performace on Different Threashold')

bar = [0.1, 0.3, 0.5, 0.7, 0.9]
F1_over_list = [0.873, 0.877, 0.878, 0.881, 0.879]
F1_ans_list = [0.838, 0.841, 0.842, 0.843,0.842]
F1_unans_list = [0.957, 0.961, 0.962, 0.965,0.966]
EM_over_list = [0.832,0.836,0.837,0.838,0.838]
EM_ans_list = [0.779,0.782,0.783,0.783,0.783]
EM_unans_list = [0.957,0.960,0.962,0.965,0.966]
axs[0].plot(bar, F1_over_list, label = 'overall')
axs[0].scatter(bar, F1_over_list)
axs[0].plot(bar, F1_ans_list, label = 'answerable')
axs[0].scatter(bar, F1_ans_list)
axs[0].plot(bar, F1_unans_list, label = 'unanswerable')
axs[0].scatter(bar, F1_unans_list)
axs[0].set_title('F1')

axs[1].plot(bar, EM_over_list, label = 'overall')
axs[1].scatter(bar, EM_over_list)
axs[1].plot(bar, EM_ans_list, label = 'answerable')
axs[1].scatter(bar, EM_ans_list)
axs[1].plot(bar, EM_unans_list, label = 'unanswerable')
axs[1].scatter(bar, EM_unans_list)
axs[1].set_title('EM')
axs[1].legend()
```






