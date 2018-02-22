# coding: utf-8

from LinearRegressionTools import linearModelBuilder,readData
import numpy as np

customers=readData('Ecommerce Customers')


customers['medium']=customers['Time on App']/customers['Time on Website']

def mediumCat (medium):
    #function to determine which medium a user used more,relative to the average
    #app:web ratio.
    mean=np.mean(customers['medium'])
    if medium>mean:
        return 'App'
    else:
        return 'Web'

customers['mediumCat']=customers['medium'].apply(mediumCat)

y=customers['Yearly Amount Spent']

X=customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
lm=linearModelBuilder(X,y)

X=customers[['Avg. Session Length','medium','Length of Membership']]
lm2=linearModelBuilder(X,y)

X=customers[['Avg. Session Length','Time on App','Length of Membership']]
lm3=linearModelBuilder(X,y)


# ** How can you interpret these coefficients? **

# The weakest correlation is between time on Website and the amount spent. Time spent on the mobile app did result in more spending with a good corelation. The average session length did affect the amount spent although it was not as influetial as the medium through which the customers interacted with the online shop. The strongest link was between the length of a customers membership and their yearly expenditure.
# 
# Customers who spent more time on the mobile app as a proportion of their time browsing the shop were more likley to spend more, despite the average time spent on the mobile app being less than half that spent on the website. The more time that the customer remained engaged with the shop, the more they were likley to spend although as we can see that most of the time was spent browsing the website and that time spent browsing the website was not related to spending this corelation must have come from the proportion of sessions that occured through the app. The customers also appear to want to spend more on the services the longer that they have been utilising them.
# 
# This indicates that your best customers are ones that have developed a relationship with the company and make more use of the app, perhaps as they already desire to spend more money.The least receptive customers are relative newcomers, potentialy drawn to the company by advertising and having a browse of the website without spending much if any money. Therefore providing rewards for loyalty would be more of an investment than rewards for newcomers.
# 
# The best performing sales interface is by far the app and the fact that the website doesn't appear to affect sales despite people spending more time on it indicates that the website could either be improved or that it is being used in a different way. For instance, sales are more corelated with time spent on the app but on average people spend less time on the app. Since people have to open a bespoke application to visit the app they will have probably opened the app with the intention to make a purchase. This indicates that perhaps when people already know what they want they spend less time shopping per amount of money spent. Customers who are less sure will probably spend more time deciding and be less likley to make a purchase, they could also be more likley to visit the website as they are not visiting the store with a purpose and have perhaps come across the store whilst browsing and following links and adverts on the web. The website is more of an advertising tool than the App as it is visited more frequently and by people that are more likley to require convincing before making a purchase. The App is more of a sales tool that enables people to make purchases more easily (potentialy before they have time to re-think their decision!). therefore, ensuring a clear path to the checkout on the app with minimal advertising and upsell incentives en-route could boost app performance and focusing advertisment on the website where the audience will need more convincing.

# **Do you think the company should focus more on their mobile app or on their website?**

# The mobile app is already proving to be an effective sales tool however as people are spending more time on the website and since time spent browsing appears to be a good indicator of sales, if the website was made as effective as the app (in terms of amount spent per amount of time using the app/website) then there is the most scope for increased sales.
# 
# Another consideration is that if the amount of time spent using the website has no impact on sales it could be that it is not as engaging as the app. Since this is the main medium through which you are interacting with your customers and that the best customers you have are those that have remained engaged for an extended period of time, making the website more engaging could help to keep customers engaged for longer thus encouraging brand loyalty and the corresponding increase in sales that has been shown to result from this.
# 
# These correlations could also be to do with how people tend to use apps and websites in different ways. Apps are generally used with more purpouse and for convinience whereas websites are often used having followed a link from an advert or simply to browse with less time constraints. This means that people are more likley to make a sale when they use the app regardless of it's quality as they are more likley to have visited it with the intention of making a purchase than when they visit the website. This therefore could simply be a reflection that customers who use the app more are also customers that want to spend more regardless of your marketing. This would therefore mean that you could benefit from investment in the app as it would be targeting a more receptive audience.

# ## Great Job!
# 
# Congrats on your contract work! The company loved the insights! Let's move on.
