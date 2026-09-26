"""
Generate synthetic email dataset for text classification
"""
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define email categories
CATEGORIES = {
    'work': [
        "Meeting scheduled for tomorrow at {}. Please confirm your attendance.",
        "Project update: {} milestone completed. Next steps discussed.",
        "Quarterly review results are attached. Please review and provide feedback.",
        "Team collaboration on {} project. Your input needed on the proposal.",
        "Budget approval required for {} initiative. Documentation attached.",
        "Performance review scheduled for {}. Prepare your self-assessment.",
        "Client presentation on {} - requesting your participation.",
        "Deadline reminder: {} deliverable due by end of week.",
        "Conference call scheduled to discuss {} strategy.",
        "Request for proposal (RFP) response needed for {} project.",
        "Annual planning session for {} department next week.",
        "Stakeholder meeting agenda for {} discussion attached.",
        "Action items from yesterday's meeting on {} project.",
        "Resource allocation for {} team - please review headcount.",
        "Compliance training mandatory for all {} employees.",
    ],
    'personal': [
        "Hey! How are you doing? Long time no see. Let's catch up soon!",
        "Thanks for the birthday wishes! Had a great celebration with family.",
        "Planning a get-together this weekend. Hope you can make it!",
        "Just wanted to check in and see how things are going with you.",
        "Remember that restaurant we talked about? Finally tried it - amazing!",
        "Can you recommend a good {} for the kids? Looking for suggestions.",
        "Family vacation photos from {} are attached. What a trip!",
        "Book club meeting next {} - have you finished reading?",
        "Thanks for helping out last week. Really appreciate it!",
        "Coffee sometime next week? Let me know when you're free.",
        "Congratulations on your new {}! So happy for you.",
        "Movie night this Friday - interested in joining us?",
        "How's your {} going? Mine has been keeping me busy!",
        "Dinner party invitation for {}. Please RSVP.",
        "Just sharing some photos from our recent {} trip!",
    ],
    'promotions': [
        "SALE: Up to 50% off on {}! Limited time offer. Shop now!",
        "Exclusive deal just for you: {} now available at discounted price.",
        "Black Friday Preview: Early access to {} deals starting tomorrow!",
        "Your favorite {} is back in stock! Get yours before they're gone.",
        "Special discount code inside: Save 30% on your next {} purchase.",
        "Flash sale alert! {} items marked down for next 24 hours only.",
        "Loyalty rewards: Earn double points on all {} purchases this month.",
        "New arrivals: Check out our latest {} collection. Free shipping!",
        "Clearance event: Final markdown on {} - up to 70% off!",
        "Subscribe and save 20% on {}. Plus free delivery on all orders.",
        "Member exclusive: Early access to {} before public launch.",
        "Bundle deal: Buy {} and get {} absolutely free!",
        "Weekend special: {} at unbeatable prices. Don't miss out!",
        "Refer a friend and both get 25% off on {}!",
        "Last chance! {} sale ends midnight tonight. Shop now!",
    ],
    'spam': [
        "You've won $1,000,000! Claim your prize now by clicking here!!!",
        "Urgent: Your account will be suspended. Verify your information immediately.",
        "Make money working from home! Earn ${} per day guaranteed!!!",
        "Singles in your area want to meet you! Click here now!!!",
        "Congratulations! You've been selected for a special {}.",
        "URGENT: IRS Tax Refund - claim your ${} now!!!",
        "Your package is waiting. Confirm shipping address immediately or will be returned.",
        "Pharmacy online - lowest prices on {} medication! No prescription needed.",
        "Get rich quick! Investment opportunity with guaranteed {}% returns!",
        "Free iPhone! You've been selected as our {}th visitor. Claim now!",
        "Weight loss miracle! Lose {} pounds in just one week!",
        "Credit card approved! ${} limit pre-approved. Apply now!!!",
        "Your computer is infected! Download our {} removal tool immediately!",
        "Nigerian prince needs your help transferring ${} million dollars.",
        "Casino online - free ${} bonus! Play now and win big!!!",
    ],
    'social': [
        "Facebook: {} commented on your post. See what they said.",
        "LinkedIn: You have {} new connection requests waiting.",
        "Twitter: {} mentioned you in a tweet. Check it out!",
        "Instagram: {} liked your recent photo. View their profile.",
        "{} shared a post that you might be interested in.",
        "Your friend {} just joined {}! Say hello.",
        "New message from {} in your {} group chat.",
        "{} tagged you in {} photos from the event.",
        "You have {} notifications from friends on {}.",
        "{} is celebrating their birthday today! Send them wishes.",
        "Your post on {} has received {} likes and {} comments.",
        "Weekly digest: Here's what you missed on {} this week.",
        "{} invited you to join their {} group.",
        "Reminder: {} event you're interested in starts tomorrow!",
        "{} started following you on {}. Follow back?",
    ],
    'finance': [
        "Your {} credit card statement is now available. Amount due: ${}.",
        "Bank alert: Transaction of ${} processed on your account.",
        "Payment confirmation: ${} received for invoice {}.",
        "Monthly statement for {} account - balance: ${}.",
        "Investment portfolio update: Your {} holdings summary for {}.",
        "Upcoming payment due: ${} for {} on {}.",
        "Wire transfer confirmation: ${} sent to {} account.",
        "Overdraft protection activated on account ending in {}.",
        "Tax documents for {} year are now available for download.",
        "Automated payment scheduled for ${} on {} has been processed.",
        "Fraud alert: Unusual activity detected on your {} card.",
        "Interest rate change notification for your {} account.",
        "Dividend payment of ${} deposited to your {} account.",
        "Annual fee of ${} charged to your {} credit card.",
        "Bill payment confirmation: ${} paid to {} successfully.",
    ],
    'newsletter': [
        "Weekly tech digest: Top {} stories you might have missed.",
        "Your daily news roundup for {} is here.",
        "Monthly newsletter: Updates on {} and upcoming events.",
        "Industry insights: Latest trends in {} sector this week.",
        "Blog update: New article on {} just published!",
        "Podcast episode: Interview with {} expert on {}.",
        "Webinar invitation: Learn about {} strategies next {}.",
        "Research report: {} market analysis for {} quarter.",
        "Community update: New features and improvements in {}.",
        "Educational series: Part {} of our {} masterclass.",
        "Case study: How {} company achieved {}% growth using {}.",
        "Product roadmap: What's coming to {} in {}.",
        "Survey invitation: Share your feedback on {}. Takes 5 minutes.",
        "Event recap: Highlights from {} conference last week.",
        "Expert roundtable: {} leaders discuss {} trends.",
    ],
}

# Filler words and variations
FILLERS = ['marketing', 'sales', 'Q3', 'Q4', '2024', '2025', 'Monday', 'Tuesday', 
           'Wednesday', 'Thursday', 'Friday', 'laptop', 'software', 'electronics',
           'clothing', 'furniture', 'books', 'courses', 'hotel', 'restaurant',
           '10 AM', '2 PM', '3:30 PM', 'product', 'service', 'digital', 'online',
           'mobile', 'cloud', 'AI', 'data', 'platform', 'summer', 'winter', 'fall',
           '100', '500', '1000', '50', '75', 'John', 'Sarah', 'Michael', 'Emily',
           'Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'YouTube', '5', '10', '15',
           'January', 'February', 'March', 'April', 'May', 'June', '1234', '5678',
           'checking', 'savings', 'January', 'tech', 'business', 'health', 'finance']

def generate_email(category):
    """Generate a single email for a given category"""
    template = random.choice(CATEGORIES[category])
    
    # Replace placeholders with fillers
    while '{}' in template:
        filler = random.choice(FILLERS)
        template = template.replace('{}', filler, 1)
    
    return template

def generate_dataset(num_samples=2000):
    """Generate complete email dataset"""
    emails = []
    labels = []
    
    # Calculate samples per category (roughly equal distribution)
    samples_per_category = num_samples // len(CATEGORIES)
    
    for category in CATEGORIES.keys():
        for _ in range(samples_per_category):
            email = generate_email(category)
            emails.append(email)
            labels.append(category)
    
    # Add remaining samples to reach exact num_samples
    remaining = num_samples - len(emails)
    for _ in range(remaining):
        category = random.choice(list(CATEGORIES.keys()))
        email = generate_email(category)
        emails.append(email)
        labels.append(category)
    
    # Create dataframe
    df = pd.DataFrame({
        'email': emails,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def save_dataset_csv(df, filename='email_dataset.csv'):
    """Save dataset to CSV without using pandas to_csv (for compatibility)"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['email', 'label'])
        for _, row in df.iterrows():
            writer.writerow([row['email'], row['label']])
    
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    try:
        print("Generating email dataset...")
        df = generate_dataset(2000)
        
        # Save to CSV
        try:
            df.to_csv('email_dataset.csv', index=False)
        except:
            # Fallback method
            save_dataset_csv(df)
        
        print(f"Dataset generated with {len(df)} samples")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"\nSample emails:")
        for idx in range(3):
            print(f"\nCategory: {df.iloc[idx]['label']}")
            print(f"Email: {df.iloc[idx]['email']}")
        
        print("\nDataset saved to email_dataset.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

