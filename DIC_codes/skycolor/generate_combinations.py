visibility=[56,131,15]
elevation=[0,90,30]
albedo=[0.0,1.0,0.5]

# for vi in range(visibility[0],visibility[1],visibility[2]):
#     cur_vis=vi+visibility[2]
#     print(cur_vis)

# for ei in range(elevation[0],elevation[1],elevation[2]):
#     cur_ele=ei+elevation[2]
#     print(cur_ele)

# for ai in range(int((albedo[1]-albedo[0])/albedo[2]+1)):
#     cur_alb=(ai*albedo[2])+albedo[0]
#     print(cur_alb)

counter=0
for vi in range(visibility[0],visibility[1],visibility[2]):
    cur_vis=vi+visibility[2]
    for ei in range(elevation[0],elevation[1],elevation[2]):
        cur_ele=ei+elevation[2]
        for ai in range(int((albedo[1]-albedo[0])/albedo[2]+1)):
            cur_alb=(ai*albedo[2])+albedo[0]

            name=str(cur_vis)+'_'+str(cur_ele)+'_'+str(int(cur_alb*10))

            print('index:',counter,'visibilty:',cur_vis,'elevation:',cur_ele,'albedo:',cur_alb,'name',name)
            counter+=1