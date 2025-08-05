#  Training Data for Custom Indian PII Redaction Model
#  Version: 1.0
#  Description: This file contains training data for a spaCy NER model
#               to recognize Indian-specific PII, including names, addresses,
#               phone numbers, Aadhaar numbers, and PAN cards.
#
# ==============================================================================

# This training data is formatted for spaCy 3.x
# Each item is a tuple: (text, {"entities": [(start_index, end_index, "LABEL")]})
TRAIN_DATA = [
    # --- Example 1: Simple Name and Email ---
    ("Contact Priya Sharma at priya.sharma@example.com for details.", {
        "entities": [(8, 20, "NAME"), (24, 49, "EMAIL")]
    }),
    
    # --- Example 2: Phone Number and Name ---
    ("Call Rohan Mehta on +91 9876543210.", {
        "entities": [(5, 16, "NAME"), (20, 33, "PHONE")]
    }),

    # --- Example 3: Full Address with PIN Code ---
    ("Ship to Flat No. 101, Sunshine Apartments, MG Road, Bangalore, 560001.", {
        "entities": [(8, 68, "ADDRESS"), (60, 68, "PINCODE")]
    }),

    # --- Example 4: Aadhaar Card Number ---
    ("His Aadhaar number is 1234 5678 9012.", {
        "entities": [(22, 36, "AADHAAR")]
    }),

    # --- Example 5: PAN Card Number ---
    ("Please verify PAN ABCDE1234F for the transaction.", {
        "entities": [(18, 28, "PAN")]
    }),

    # --- Example 6: Complex Sentence with Multiple Entities ---
    ("Anjali Gupta (email: anjali.g@email.com, phone: 9988776655) lives in Mumbai 400053.", {
        "entities": [(0, 12, "NAME"), (21, 42, "EMAIL"), (51, 61, "PHONE"), (72, 86, "ADDRESS"), (80, 86, "PINCODE")]
    }),

    # --- Example 7: Bank Account Number ---
    ("My savings account number is 01234567890.", {
        "entities": [(29, 40, "ACCOUNT_NUMBER")]
    }),

    # --- Example 8: Address with Landmark ---
    ("Address: 12/3, Main Street, Near City Hospital, Pune - 411004, Maharashtra.", {
        "entities": [(9, 73, "ADDRESS"), (49, 57, "PINCODE")]
    }),

    # --- Example 9: Name and Phone without +91 prefix ---
    ("For support, contact Vikram Singh at 8877665544.", {
        "entities": [(23, 35, "NAME"), (39, 49, "PHONE")]
    }),

    # --- Example 10: Email in a different context ---
    ("Send the invoice to accounts-payable@mycompany.co.in", {
        "entities": [(20, 53, "EMAIL")]
    }),
    
    # --- Example 11: Another Aadhaar format (no spaces) ---
    ("The beneficiary's Aadhaar is 987654321098.", {
        "entities": [(28, 40, "AADHAAR")]
    }),

    # --- Example 12: Another PAN format ---
    ("Her PAN is FGHIJ5678K.", {
        "entities": [(11, 21, "PAN")]
    }),

    # --- Example 13: Multi-line address style ---
    ("Deliver to: Sameer Khan, 45, Rose Villa, Bandra West, Mumbai, 400050.", {
        "entities": [(12, 23, "NAME"), (25, 70, "ADDRESS"), (64, 70, "PINCODE")]
    }),

    # --- Example 14: Name with middle name ---
    ("The application was submitted by Arjun Kumar Sharma.", {
        "entities": [(32, 51, "NAME")]
    }),

    # --- Example 15: Address in a sentence ---
    ("He moved to 15B, Sector 18, Noida, Uttar Pradesh 201301 last year.", {
        "entities": [(12, 59, "ADDRESS"), (53, 59, "PINCODE")]
    }),

    # --- Example 16: Another bank account number ---
    ("Please credit salary to A/C 50100123456789.", {
        "entities": [(27, 41, "ACCOUNT_NUMBER")]
    }),

    # --- Example 17: Name and Email together ---
    ("Isha Patel's email is isha.p@provider.net.", {
        "entities": [(0, 10, "NAME"), (22, 42, "EMAIL")]
    }),

    # --- Example 18: Phone number with dashes ---
    ("You can reach me at 91-999-888-7777.", {
        "entities": [(21, 36, "PHONE")]
    }),

    # --- Example 19: South Indian Address ---
    ("Resident of No. 5, 3rd Cross, Indiranagar, Chennai, Tamil Nadu - 600020.", {
        "entities": [(12, 74, "ADDRESS"), (68, 74, "PINCODE")]
    }),

    # --- Example 20: Another PAN example ---
    ("The company PAN is AABCD2345E.", {
        "entities": [(18, 28, "PAN")]
    }),

    # --- Example 21: Another Aadhaar example ---
    ("Link your Aadhaar: 3344 5566 7788.", {
        "entities": [(20, 34, "AADHAAR")]
    }),

    # --- Example 22: Address in Kolkata ---
    ("Find us at 21, Park Street, Area, Kolkata, West Bengal 700016.", {
        "entities": [(11, 62, "ADDRESS"), (56, 62, "PINCODE")]
    }),

    # --- Example 23: Simple Name and Phone ---
    ("My name is Alok Verma and my number is 9811122233.", {
        "entities": [(11, 21, "NAME"), (40, 50, "PHONE")]
    }),

    # --- Example 24: Formal Account Number ---
    ("Reference transaction for Account Number: 987654321012345.", {
        "entities": [(39, 54, "ACCOUNT_NUMBER")]
    }),
    
    # --- Example 25: Name with initials ---
    ("The package for S. K. Menon is here.", {
        "entities": [(16, 26, "NAME")]
    }),

    # --- NEW EXAMPLES START HERE ---

    # --- Example 26: Name with prefix and professional email ---
    ("Dr. Rina Kapoor can be reached at dr.rina.k@clinic.in.", {"entities": [(0, 15, "NAME"), (33, 53, "EMAIL")]}),

    # --- Example 27: Address with C/O (Care Of) ---
    ("Forward the documents to Mr. Suresh Iyer, C/O Ramnath Iyer, 10B, Ganga Complex, Hyderabad 500081.", {"entities": [(26, 41, "NAME"), (43, 95, "ADDRESS"), (89, 95, "PINCODE")]}),

    # --- Example 28: Phone with parentheses ---
    ("My mobile is (91) 8765432109.", {"entities": [(13, 28, "PHONE")]}),

    # --- Example 29: Alphanumeric Account Number ---
    ("The patient's file number is PN-567890.", {"entities": [(29, 38, "ACCOUNT_NUMBER")]}),

    # --- Example 30: PAN in a different context ---
    ("PAN details: ZYXWV9876U. Please confirm.", {"entities": [(13, 23, "PAN")]}),

    # --- Example 31: Name and email in parentheses ---
    ("Send queries to Neha Singh (neha.s@corporate.com).", {"entities": [(17, 27, "NAME"), (29, 50, "EMAIL")]}),

    # --- Example 32: Aadhaar in a sentence ---
    ("Her Aadhaar card 4567 1234 8765 was used for KYC.", {"entities": [(18, 32, "AADHAAR")]}),

    # --- Example 33: Delhi Address ---
    ("The delivery address is 7, Shakti Nagar, GT Karnal Road, Delhi - 110007.", {"entities": [(24, 76, "ADDRESS"), (68, 76, "PINCODE")]}),

    # --- Example 34: Informal phone label ---
    ("Contact person: Deepak Kumar, phone no. 9000011111.", {"entities": [(16, 28, "NAME"), (39, 49, "PHONE")]}),

    # --- Example 35: Account number with prefix and dashes ---
    ("Please make the payment to SB/0012-3456-7890.", {"entities": [(28, 44, "ACCOUNT_NUMBER")]}),

    # --- Example 36: Passport Number ---
    ("My passport number is M1234567.", {"entities": [(22, 30, "PASSPORT_ID")]}),

    # --- Example 37: Voter ID Card ---
    ("Voter ID card number is XYZ1234567.", {"entities": [(24, 34, "VOTER_ID")]}),

    # --- Example 38: Commercial Address ---
    ("Meet me at 123, Commercial Street, Shivaji Nagar, Bangalore, 560001.", {"entities": [(12, 70, "ADDRESS"), (64, 70, "PINCODE")]}),

    # --- Example 39: Generic support email ---
    ("For grievances, mail help@customersupport.co.in", {"entities": [(21, 48, "EMAIL")]}),

    # --- Example 40: Name with title ---
    ("The account for Mrs. Geeta Rao is inactive.", {"entities": [(16, 30, "NAME")]}),

    # --- Example 41: Aadhaar with label ---
    ("Aadhaar: 556677889900", {"entities": [(9, 21, "AADHAAR")]}),

    # --- Example 42: Informal phone context ---
    ("Call me on my other number: 7776665554.", {"entities": [(28, 38, "PHONE")]}),

    # --- Example 43: PAN with label ---
    ("PAN: LMNOP1234Q", {"entities": [(5, 15, "PAN")]}),

    # --- Example 44: Simple name in a sentence ---
    ("The cheque was issued to Rohan Desai.", {"entities": [(25, 36, "NAME")]}),

    # --- Example 45: Gurgaon Address ---
    ("Our office is located in Cyber City, Building 5, Gurgaon, Haryana 122002.", {"entities": [(26, 80, "ADDRESS"), (74, 80, "PINCODE")]}),

    # --- Example 46: Beneficiary with Account Number ---
    ("The beneficiary is Kavita Nair, A/C: 22334455667.", {"entities": [(19, 30, "NAME"), (37, 48, "ACCOUNT_NUMBER")]}),

    # --- Example 47: Email change notification ---
    ("My email has changed to new.address@server.org.", {"entities": [(24, 49, "EMAIL")]}),

    # --- Example 48: Applicant details with labels ---
    ("Applicant's name: Aarav Patel. Phone: +91-8888877777.", {"entities": [(19, 30, "NAME"), (39, 53, "PHONE")]}),

    # --- Example 49: Salt Lake, Kolkata Address ---
    ("The address is 4th Floor, Tech Tower, Salt Lake, Kolkata 700091.", {"entities": [(15, 68, "ADDRESS"), (62, 68, "PINCODE")]}),

    # --- Example 50: Voter ID in a sentence ---
    ("His Voter ID is JKL9876543.", {"entities": [(16, 26, "VOTER_ID")]}),

    # --- Example 51: Passport number with label ---
    ("Passport No.: R8765432", {"entities": [(14, 22, "PASSPORT_ID")]}),

    # --- Example 52: Phone number embedded in text ---
    ("Please note down my number 9123456789 for future reference.", {"entities": [(28, 38, "PHONE")]}),

    # --- Example 53: Name with middle initial ---
    ("The policy holder is Mr. Vijay K. Reddy.", {"entities": [(21, 38, "NAME")]}),

    # --- Example 54: PAN card in a sentence ---
    ("My PAN card number is QRSTU5432P.", {"entities": [(22, 32, "PAN")]}),

    # --- Example 55: Thane Address ---
    ("The package should be sent to 9B, Hiranandani Estate, Thane West, 400607.", {"entities": [(31, 78, "ADDRESS"), (72, 78, "PINCODE")]}),

    # --- Example 56: Phone linked to Aadhaar ---
    ("My Aadhaar is linked to phone 9999900000.", {"entities": [(31, 41, "PHONE")]}),

    # --- Example 57: Email as User ID ---
    ("User ID: anita.1990@webmail.com", {"entities": [(9, 32, "EMAIL")]}),

    # --- Example 58: Current Account with prefix and dashes ---
    ("The current account is CA/9876-5432-1098.", {"entities": [(24, 40, "ACCOUNT_NUMBER")]}),

    # --- Example 59: Name at the end of a sentence ---
    ("This is regarding the application of Fatima Sheikh.", {"entities": [(36, 50, "NAME")]}),

    # --- Example 60: Chandigarh Address ---
    ("Find our location at Plot 42, Sector 29, Chandigarh 160030.", {"entities": [(22, 64, "ADDRESS"), (58, 64, "PINCODE")]}),

    # --- Example 61: Simple phone number announcement ---
    ("My new number is 9821098210.", {"entities": [(17, 27, "PHONE")]}),

    # --- Example 62: Aadhaar validation context ---
    ("The Aadhaar number 876512340987 is not valid.", {"entities": [(19, 31, "AADHAAR")]}),

    # --- Example 63: Name with title in a sentence ---
    ("The complaint was filed by Dr. Anand Joshi.", {"entities": [(28, 43, "NAME")]}),

    # --- Example 64: Passport ID in a sentence ---
    ("Her passport ID is T12345678.", {"entities": [(19, 28, "PASSPORT_ID")]}),

    # --- Example 65: Voter ID with context ---
    ("The Voter ID for the head of family is ABC0123456.", {"entities": [(36, 46, "VOTER_ID")]}),

    # --- Example 66: Career-related email ---
    ("Mail your resume to careers@bigcorp.net", {"entities": [(21, 40, "EMAIL")]}),

    # --- Example 67: Chennai Address with landmark ---
    ("Address: House No. 55, Anna Salai, T. Nagar, Chennai - 600017.", {"entities": [(9, 64, "ADDRESS"), (56, 64, "PINCODE")]}),

    # --- Example 68: Account number in a transaction sentence ---
    ("The amount was transferred to account 112233445566.", {"entities": [(37, 49, "ACCOUNT_NUMBER")]}),

    # --- Example 69: Multiple PII - Name and PAN ---
    ("This is to confirm that Smita Patil's PAN is BCDEF4321G.", {"entities": [(26, 37, "NAME"), (45, 55, "PAN")]}),

    # --- Example 70: Multiple PII - Name and Phone with dashes ---
    ("Please call Ishan Malhotra at +91-98765-12345 for assistance.", {"entities": [(12, 27, "NAME"), (31, 46, "PHONE")]}),

    # --- Example 71: Pune Address ---
    ("My permanent address is 101, Royal Palms, Koregaon Park, Pune 411001.", {"entities": [(24, 73, "ADDRESS"), (67, 73, "PINCODE")]}),

    # --- Example 72: Aadhaar with dashes ---
    ("The Aadhaar 7766-8899-0011 is not yet verified.", {"entities": [(11, 25, "AADHAAR")]}),

    # --- Example 73: Complex South Indian Name ---
    ("The client, Mr. R. Balasubramanian, will join shortly.", {"entities": [(13, 36, "NAME")]}),

    # --- Example 74: Lost Voter ID context ---
    ("My Voter ID is lost, the number was GHI8765432.", {"entities": [(36, 46, "VOTER_ID")]}),

    # --- Example 75: Passport number in context ---
    ("His passport, P9876543, expires next year.", {"entities": [(14, 22, "PASSPORT_ID")]}),
    
    # --- Example 76: US Address with ZIP Code ---
    ("Mail checks to 123 Main St, Anytown, CA 90210, USA.", {"entities": [(15, 49, "ADDRESS"), (38, 43, "ZIPCODE")]}),

    # --- Example 77: UK Phone Number and Name ---
    ("Contact Mr. Alistair Finch at +44 20 7946 0958 for a quote.", {"entities": [(8, 26, "NAME"), (30, 47, "PHONE")]}),

    # --- Example 78: University Email Address ---
    ("The research paper was submitted by Prof. Evelyn Reed via ereed@university.edu.", {"entities": [(35, 53, "NAME"), (58, 78, "EMAIL")]}),

    # --- Example 79: Canadian Address with Postal Code ---
    ("Shipment destination: 455 Front St W, Toronto, ON M5V 2T6, Canada.", {"entities": [(22, 70, "ADDRESS"), (57, 64, "ZIPCODE")]}),

    # --- Example 80: SWIFT Code for International Transfer ---
    ("For wire transfers, use SWIFT code CITIUS33 and account 9870001234.", {"entities": [(28, 38, "SWIFT_CODE"), (50, 60, "ACCOUNT_NUMBER")]}),

    # --- Example 81: Formal Name with Title (CEO) ---
    ("A statement was released by the CEO, Jonathan Hayes.", {"entities": [(34, 49, "NAME")]}),

    # --- Example 82: Australian Phone Number ---
    ("Our Sydney office can be reached at +61 2 9876 5432.", {"entities": [(36, 52, "PHONE")]}),
    
    # --- Example 83: Generic ID Number ---
    ("Please reference your Membership ID: M-2024-8877.", {"entities": [(32, 45, "ID_NUMBER")]}),

    # --- Example 84: Non-profit Email Domain ---
    ("Donations can be discussed with info@charityfoundation.org.", {"entities": [(31, 58, "EMAIL")]}),

    # --- Example 85: German Address ---
    ("The package is for Klaus Schmidt, Museumstr. 20, 10117 Berlin, Germany.", {"entities": [(20, 34, "NAME"), (36, 76, "ADDRESS"), (56, 61, "ZIPCODE")]}),

    # --- Example 86: US Phone Number with Extension ---
    ("Call our main line at 1-800-555-0199 ext. 1234.", {"entities": [(23, 46, "PHONE")]}),

    # --- Example 87: Government Email Domain ---
    ("Submit all forms to compliance@federalagency.gov.", {"entities": [(21, 49, "EMAIL")]}),

    # --- Example 88: Full UK Address ---
    ("Send invoice to 221B Baker Street, Marylebone, London NW1 6XE, UK.", {"entities": [(16, 69, "ADDRESS"), (56, 63, "ZIPCODE")]}),

    # --- Example 89: IBAN (International Bank Account Number) ---
    ("The payment must be sent to IBAN GB29 NWBK 6016 1331 9268 19.", {"entities": [(29, 61, "ACCOUNT_NUMBER")]}),

    # --- Example 90: Name with Post-nominal Letters ---
    ("The report was authored by Dr. Kenji Tanaka, PhD.", {"entities": [(28, 50, "NAME")]}),

    # --- Example 91: Formal Request with Name and Email ---
    ("Please direct your questions to Olivia Chen at o.chen@globaltech.ca.", {"entities": [(31, 42, "NAME"), (46, 67, "EMAIL")]}),

    # --- Example 92: French Address ---
    ("Recipient: Pierre Dubois, 18 Rue de la Paix, 75002 Paris, France.", {"entities": [(11, 24, "NAME"), (26, 69, "ADDRESS"), (50, 55, "ZIPCODE")]}),

    # --- Example 93: Customer ID ---
    ("For faster service, please provide your Customer ID: CUST-901-2024.", {"entities": [(43, 60, "ID_NUMBER")]}),

    # --- Example 94: Name and Passport ID in a single sentence ---
    ("The visa for applicant Maria Garcia (Passport A98765432) is approved.", {"entities": [(25, 37, "NAME"), (48, 57, "PASSPORT_ID")]}),

    # --- Example 95: UK Mobile Number ---
    ("Her mobile contact is +44 7700 900123.", {"entities": [(22, 38, "PHONE")]}),

    # --- Example 96: Japanese Address ---
    ("Deliver to: 1-1-2 Oshiage, Sumida-ku, Tokyo 131-8634, Japan.", {"entities": [(12, 63, "ADDRESS"), (45, 53, "ZIPCODE")]}),

    # --- Example 97: Email in a legal document ---
    ("All legal notices should be sent to legal.dept@megacorp.com.", {"entities": [(36, 61, "EMAIL")]}),

    # --- Example 98: Multiple International Phones ---
    ("Contact us in London (+44 20 8123 4567) or New York (+1 212 345 6789).", {"entities": [(22, 40, "PHONE"), (53, 69, "PHONE")]}),

    # --- Example 99: SWIFT and IBAN together ---
    ("Use SWIFT: DEUTDEFF and IBAN: DE89 3704 0044 0532 0130 00.", {"entities": [(9, 19, "SWIFT_CODE"), (29, 61, "ACCOUNT_NUMBER")]}),

    # --- Example 100: Name with multiple titles ---
    ("The keynote was given by The Rt. Hon. Lord David Richards.", {"entities": [(27, 55, "NAME")]}),

    # --- Example 101: Australian Address ---
    ("Return address: PO Box 123, Sydney, NSW 2001, Australia.", {"entities": [(16, 58, "ADDRESS"), (40, 48, "ZIPCODE")]}),

    # --- Example 102: Employee ID in context ---
    ("The access card for employee E45-B321 is now active.", {"entities": [(29, 37, "ID_NUMBER")]}),

    # --- Example 103: Formal email signature ---
    ("Sincerely, James T. Kirk, CEO. Email: j.kirk@enterprise.net", {"entities": [(12, 27, "NAME"), (40, 63, "EMAIL")]}),

    # --- Example 104: US address with apartment number ---
    ("Send to: 456 Oak Avenue, Apt 7B, Springfield, IL 62704.", {"entities": [(9, 57, "ADDRESS"), (51, 56, "ZIPCODE")]}),

    # --- Example 105: UK bank sort code and account number ---
    ("Pay to Sort Code 12-34-56, Account No. 98765432.", {"entities": [(17, 26, "ACCOUNT_NUMBER"), (40, 50, "ACCOUNT_NUMBER")]}),

    # --- Example 106: Name and Canadian Phone ---
    ("For booking, call Ms. Emily Tremblay at +1 (604) 555-0123.", {"entities": [(20, 36, "NAME"), (40, 58, "PHONE")]}),

    # --- Example 107: Generic support email address ---
    ("For technical support, please email support@techsolutions.io.", {"entities": [(34, 59, "EMAIL")]}),

    # --- Example 108: Passport number in a form-like structure ---
    ("Field 7: Passport Number. Entry: YZ9876543.", {"entities": [(34, 43, "PASSPORT_ID")]}),

    # --- Example 109: Formal name with middle name and title ---
    ("The project was approved by Director Michael B. Jordan.", {"entities": [(31, 54, "NAME")]}),

    # --- Example 110: Another SWIFT code example ---
    ("Our bank's SWIFT is HSBCHKHHHKH.", {"entities": [(20, 31, "SWIFT_CODE")]}),

    # --- Example 111: Email address with a number ---
    ("Contact sales at sales2024@online-retail.com for deals.", {"entities": [(18, 46, "EMAIL")]}),

    # --- Example 112: Full Canadian address ---
    ("Mail to: 1000, rue De la Gauchetière Ouest, Montréal, QC H3B 4W5.", {"entities": [(9, 68, "ADDRESS"), (61, 68, "ZIPCODE")]}),

    # --- Example 113: Name and Employee ID together ---
    ("The file for Chloe O'Brian (ID: 7G-8812) is attached.", {"entities": [(12, 26, "NAME"), (32, 39, "ID_NUMBER")]}),

    # --- Example 114: Phone number with country code and no spaces ---
    ("You can reach our global hotline at +442079460958.", {"entities": [(35, 49, "PHONE")]}),

    # --- Example 115: Another IBAN example ---
    ("Please use IBAN FR76 3000 6000 0112 3456 7890 189 for payment.", {"entities": [(12, 49, "ACCOUNT_NUMBER")]}),

    # --- Example 116: Formal name in a list ---
    ("Attendees: 1. Dr. Aris Thorne 2. Ms. Lena Petrova", {"entities": [(16, 30, "NAME"), (36, 51, "NAME")]}),

    # --- Example 117: US address with P.O. Box ---
    ("Mailing address: P.O. Box 9876, Denver, CO 80201.", {"entities": [(17, 52, "ADDRESS"), (46, 51, "ZIPCODE")]}),

    # --- Example 118: Email with a subdomain ---
    ("Contact the registrar at admissions@dept.university.edu.", {"entities": [(26, 57, "EMAIL")]}),

    # --- Example 119: Passport and name in reverse order ---
    ("Passport K12345678 belongs to Mr. Samuel Chen.", {"entities": [(9, 18, "PASSPORT_ID"), (32, 47, "NAME")]}),

    # --- Example 120: Phone number with dots as separators ---
    ("Our fax number is +1.212.555.0123.", {"entities": [(18, 33, "PHONE")]}),

    # --- Example 121: Full name and title in a formal context ---
    ("The ceremony was led by Professor Albus Dumbledore.", {"entities": [(25, 50, "NAME")]}),

    # --- Example 122: Another US address format ---
    ("Find us at 1 Infinite Loop, Cupertino, CA 95014.", {"entities": [(11, 49, "ADDRESS"), (43, 48, "ZIPCODE")]}),

    # --- Example 123: Simple name and international phone ---
    ("Call Omar Hassan at +20 2 2555 0199 for information.", {"entities": [(5, 16, "NAME"), (20, 36, "PHONE")]}),

    # --- Example 124: Invoice number as an ID ---
    ("Please reference Invoice #INV-2024-001 on all payments.", {"entities": [(22, 37, "ID_NUMBER")]}),

    # --- Example 125: Final example with multiple entities ---
    ("Agent 007, James Bond, can be reached at j.bond@mi6.gov.uk or via his passport ZM987654.", {"entities": [(12, 22, "NAME"), (43, 62, "EMAIL"), (76, 85, "PASSPORT_ID")]}),
    
    
    # --- NEW COMPLIANCE-SPECIFIC EXAMPLES (GDPR, HIPAA, DPDP) ---

    # --- 1. GDPR (EU General Data Protection Regulation) Examples ---
    ("User from IP address 192.168.1.1 accessed the portal.", {"entities": [(21, 32, "IP_ADDRESS")]}),
    ("The tracking cookie uses Device ID 550e8400-e29b-41d4-a716-446655440000.", {"entities": [(32, 68, "DEVICE_ID")]}),
    ("Data subject request from lars.johansson@email.se for all his data.", {"entities": [(26, 49, "EMAIL"), (13, 27, "NAME")]}),
    ("The user's location is Berlin, Germany, based on IP 89.160.20.12.", {"entities": [(24, 40, "ADDRESS"), (52, 64, "IP_ADDRESS")]}),
    ("Biometric login failed for user fingerprint hash F3-A4-C1-B9.", {"entities": [(42, 55, "BIOMETRIC_DATA")]}),
    ("Contact form submitted by François Dubois from 18 Rue de la Paix, Paris.", {"entities": [(26, 41, "NAME"), (47, 74, "ADDRESS")]}),
    ("Her advertising ID is idfa-a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d.", {"entities": [(24, 63, "DEVICE_ID")]}),
    ("The user at 2a02:c7f:d246:2100::1 requested data erasure.", {"entities": [(12, 36, "IP_ADDRESS")]}),
    ("Genetic data sample G-987-ACGT was processed on 2024-07-21.", {"entities": [(20, 31, "BIOMETRIC_DATA"), (49, 59, "DATE")]}),
    ("Marta Kowalski (m.kowalski@provider.pl) has withdrawn her consent.", {"entities": [(0, 14, "NAME"), (16, 39, "EMAIL")]}),
    ("Login attempt from IP 212.58.244.70, which maps to London.", {"entities": [(21, 34, "IP_ADDRESS"), (51, 57, "ADDRESS")]}),
    ("User sofia.rossi@email.it has requested their data portability.", {"entities": [(5, 25, "EMAIL")]}),
    ("The retinal scan data for subject E-007 is encrypted.", {"entities": [(4, 21, "BIOMETRIC_DATA")]}),
    ("Device fingerprint: 2c54f5d3e9d3e9c3c81c1c1d8e8e8e8e.", {"entities": [(20, 52, "DEVICE_ID")]}),
    ("Data processing agreement for Hans Müller of Munich.", {"entities": [(30, 41, "NAME"), (45, 51, "ADDRESS")]}),
    ("Unusual activity detected from IP 46.5.9.100 in Amsterdam.", {"entities": [(31, 41, "IP_ADDRESS"), (45, 54, "ADDRESS")]}),
    ("User ID 98765 is associated with device 123e4567-e89b-12d3-a456-426614174000.", {"entities": [(8, 13, "ID_NUMBER"), (36, 72, "DEVICE_ID")]}),
    ("Consent for marketing emails was revoked by elena.gomez@email.es.", {"entities": [(42, 62, "EMAIL")]}),
    ("Voiceprint analysis V-A-X-9-8-7 was used for authentication.", {"entities": [(20, 35, "BIOMETRIC_DATA")]}),
    ("Data controller is located at 10 Rue de la Liberté, Luxembourg.", {"entities": [(29, 64, "ADDRESS")]}),
    ("The IP address 94.126.16.88 is blocked.", {"entities": [(15, 27, "IP_ADDRESS")]}),
    ("User Jean-Pierre (jp@email.fr) accessed the file.", {"entities": [(5, 16, "NAME"), (18, 29, "EMAIL")]}),
    ("Facial recognition data F-ID-90210 is stored securely.", {"entities": [(24, 33, "BIOMETRIC_DATA")]}),
    ("The user's registered address is Via Roma 1, Milan, Italy.", {"entities": [(30, 58, "ADDRESS")]}),
    ("A data breach notification was sent to anders.nielsen@email.dk.", {"entities": [(38, 64, "EMAIL")]}),
    ("The user's browser agent reported device ID abc-123-def-456.", {"entities": [(39, 55, "DEVICE_ID")]}),
    ("Access log shows IP 195.20.5.10 at timestamp 1660573192.", {"entities": [(18, 29, "IP_ADDRESS")]}),
    ("Data of Mr. Lukas Novak must be anonymized.", {"entities": [(11, 22, "NAME")]}),
    ("User's location determined as Dublin from IP 176.61.134.130.", {"entities": [(30, 36, "ADDRESS"), (45, 59, "IP_ADDRESS")]}),
    ("The genetic sequence GATTACA is under strict access control.", {"entities": [(22, 29, "BIOMETRIC_DATA")]}),
    ("Session cookie linked to device id: aabbcc-ddeeff-001122.", {"entities": [(32, 55, "DEVICE_ID")]}),
    ("Right to be forgotten request from maria.silva@email.pt.", {"entities": [(34, 55, "EMAIL")]}),
    ("The user's IP is 2001:0db8:85a3:0000:0000:8a2e:0370:7334.", {"entities": [(16, 55, "IP_ADDRESS")]}),
    ("Keystroke dynamics profile KDP-9981 is a form of biometric data.", {"entities": [(27, 35, "BIOMETRIC_DATA")]}),
    ("User from Warsaw, Poland (IP: 83.12.0.0) visited the site.", {"entities": [(10, 25, "ADDRESS"), (31, 40, "IP_ADDRESS")]}),

    # --- 2. HIPAA (US Health Insurance Portability and Accountability Act) Examples ---
    ("Patient John Smith (MRN: 12345) was admitted on 08-15-2024.", {"entities": [(8, 18, "NAME"), (25, 30, "MRN"), (48, 58, "DATE")]}),
    ("Prescription for Jane Doe: 20mg of Lisinopril daily.", {"entities": [(18, 26, "NAME"), (33, 52, "MEDICATION")]}),
    ("Discharge summary for patient, DOB: 01/20/1985.", {"entities": [(35, 46, "DATE")]}),
    ("Health Plan ID for Robert Johnson is HPN-AETNA-98765.", {"entities": [(15, 29, "NAME"), (33, 52, "HEALTH_PLAN_ID")]}),
    ("Diagnosis: Type 2 Diabetes. Patient advised on diet.", {"entities": [(11, 28, "DIAGNOSIS")]}),
    ("Dr. Emily White's patient, Mary-Anne, has an appointment on 12/25/2025.", {"entities": [(3, 18, "NAME"), (29, 38, "NAME"), (62, 72, "DATE")]}),
    ("Medical Record Number 987654 belongs to Mr. Davis.", {"entities": [(22, 28, "MRN"), (42, 51, "NAME")]}),
    ("The patient was prescribed Metformin 500mg twice a day.", {"entities": [(26, 43, "MEDICATION")]}),
    ("Admission date: 09/01/2024. Discharge date: 09/08/2024.", {"entities": [(16, 26, "DATE"), (45, 55, "DATE")]}),
    ("Patient's insurance is Blue Cross, ID: BCBS-12345-01.", {"entities": [(35, 53, "HEALTH_PLAN_ID")]}),
    ("Consultation notes for Susan Miller regarding her hypertension.", {"entities": [(24, 36, "NAME"), (52, 64, "DIAGNOSIS")]}),
    ("Lab results for MRN 555-666-777 are pending.", {"entities": [(17, 28, "MRN")]}),
    ("The patient, born 03-MAR-1990, is allergic to Penicillin.", {"entities": [(19, 30, "DATE"), (48, 58, "MEDICATION")]}),
    ("Referral to Dr. Charles Brown at Mercy General Hospital.", {"entities": [(13, 30, "NAME"), (34, 56, "ADDRESS")]}),
    ("Patient's health plan is UnitedHealthcare, policy #UHC12345.", {"entities": [(40, 50, "HEALTH_PLAN_ID")]}),
    ("Initial diagnosis is acute bronchitis.", {"entities": [(21, 37, "DIAGNOSIS")]}),
    ("Please administer 10 units of Insulin before meals.", {"entities": [(30, 37, "MEDICATION")]}),
    ("Patient DOB is 11/30/1975, MRN is 78901.", {"entities": [(15, 25, "DATE"), (32, 37, "MRN")]}),
    ("Surgical procedure performed on 10-02-2024 by Dr. Davis.", {"entities": [(31, 41, "DATE"), (45, 54, "NAME")]}),
    ("The member ID for Cigna is 789XYZ123.", {"entities": [(27, 36, "HEALTH_PLAN_ID")]}),
    ("Patient history shows a prior diagnosis of asthma.", {"entities": [(40, 46, "DIAGNOSIS")]}),
    ("Medication list includes Atorvastatin 40mg.", {"entities": [(24, 42, "MEDICATION")]}),
    ("This chart belongs to Michael P. Jones, born on May 5, 1960.", {"entities": [(21, 39, "NAME"), (49, 60, "DATE")]}),
    ("The patient's MRN is 001-002-003.", {"entities": [(19, 30, "MRN")]}),
    ("Allergy to Ibuprofen noted in chart.", {"entities": [(12, 21, "MEDICATION")]}),
    ("Appointment scheduled with Dr. Patel on 01/15/2025.", {"entities": [(28, 37, "NAME"), (41, 51, "DATE")]}),
    ("The patient has a family history of coronary artery disease.", {"entities": [(34, 59, "DIAGNOSIS")]}),
    ("Insurance provider: Humana, Group No. GRP-54321.", {"entities": [(34, 50, "HEALTH_PLAN_ID")]}),
    ("Patient was discharged on June 1st, 2023.", {"entities": [(26, 41, "DATE")]}),
    ("Take two tablets of Amoxicillin 500mg every 12 hours.", {"entities": [(20, 37, "MEDICATION")]}),
    ("MRN: 998877 for patient Williams, Chris.", {"entities": [(5, 11, "MRN"), (23, 38, "NAME")]}),
    ("Date of birth: July 4, 1976.", {"entities": [(17, 30, "DATE")]}),
    ("The patient shows symptoms of pneumonia.", {"entities": [(30, 39, "DIAGNOSIS")]}),
    ("Her health insurance ID is ANTHEM-9988-77.", {"entities": [(27, 44, "HEALTH_PLAN_ID")]}),
    ("Dr. Jessica Chen prescribed Warfarin.", {"entities": [(3, 18, "NAME"), (30, 38, "MEDICATION")]}),

    # --- 3. DPDP (India - Digital Personal Data Protection Act) Examples ---
    ("User profile for Aarav Sharma. Gender: Male. Religion: Hindu.", {"entities": [(17, 29, "NAME"), (38, 42, "GENDER"), (54, 59, "RELIGION")]}),
    ("KYC verification for Fatima Sheikh failed. Aadhaar: 9876 1234 5678.", {"entities": [(21, 34, "NAME"), (52, 66, "AADHAAR")]}),
    ("The data fiduciary is BigTech Pvt Ltd, located in Mumbai.", {"entities": [(49, 55, "ADDRESS")]}),
    ("Consent for data processing was given by Riya Singh.", {"entities": [(41, 51, "NAME")]}),
    ("User's registered mobile is 9090909090 and PAN is GHIJK1234L.", {"entities": [(29, 39, "PHONE"), (50, 60, "PAN")]}),
    ("The user's caste is listed as Other Backward Class.", {"entities": [(30, 50, "RELIGION")]}), # Caste can be considered sensitive personal data
    ("Data principal Sameer Ali has requested data correction.", {"entities": [(16, 26, "NAME")]}),
    ("User's gender is Female, and she is a resident of Delhi.", {"entities": [(16, 22, "GENDER"), (51, 56, "ADDRESS")]}),
    ("The Aadhaar number 1122-3344-5566 is linked to this account.", {"entities": [(19, 33, "AADHAAR")]}),
    ("A user, whose religion is Sikh, filed a complaint.", {"entities": [(28, 32, "RELIGION")]}),
    ("PAN Card: LMNOP5432Q for applicant Priya Kumari.", {"entities": [(10, 20, "PAN"), (35, 47, "NAME")]}),
    ("The user identified as belonging to the Christian community.", {"entities": [(40, 49, "RELIGION")]}),
    ("Data processing notice sent to anita.d@email.co.in.", {"entities": [(31, 51, "EMAIL")]}),
    ("The user's gender is not specified in the profile.", {"entities": []}), # Negative example
    ("Aadhaar details of Mr. Vijay Kumar are confidential.", {"entities": [(19, 31, "NAME")]}),
    ("The user's phone number is +91-800-700-6000.", {"entities": [(27, 42, "PHONE")]}),
    ("The user's religious preference is listed as Islam.", {"entities": [(43, 48, "RELIGION")]}),
    ("PAN details for Isha Foundation are not personal data.", {"entities": []}), # Negative example
    ("The user's gender is Male.", {"entities": [(20, 24, "GENDER")]}),
    ("This data belongs to Arjun, whose Aadhaar is 9988 7766 5544.", {"entities": [(21, 26, "NAME"), (40, 54, "AADHAAR")]}),
    ("The user is a follower of Jainism.", {"entities": [(27, 34, "RELIGION")]}),
    ("Contact details of Neha Gupta: 9876556789.", {"entities": [(19, 29, "NAME"), (31, 41, "PHONE")]}),
    ("The data principal's gender is Transgender.", {"entities": [(31, 42, "GENDER")]}),
    ("PAN of the company is AABBC1234D.", {"entities": [(21, 31, "PAN")]}),
    ("The user's faith is Buddhism.", {"entities": [(22, 30, "RELIGION")]}),
    ("Aadhaar number cannot be used for marking attendance.", {"entities": []}), # Contextual negative
    ("The user's name is Kabir Singh and he is a Muslim.", {"entities": [(19, 30, "NAME"), (42, 48, "RELIGION")]}),
    ("Phone number 9112233445 is registered with the app.", {"entities": [(13, 23, "PHONE")]}),
    ("Gender: F. Name: Sunita Sharma.", {"entities": [(8, 9, "GENDER"), (17, 30, "NAME")]}),
    ("The user does not follow any particular religion.", {"entities": []}),
    ("PAN verification is mandatory for Mr. Alok Jain.", {"entities": [(37, 47, "NAME")]}),
    ("The user's Aadhaar number is masked as XXXX XXXX 1234.", {"entities": []}), # Negative example
    ("The user's religious beliefs are not collected.", {"entities": []}),
    ("User's gender: Other. User's name: Alex.", {"entities": [(15, 20, "GENDER"), (34, 38, "NAME")]}),
    ("PAN: ZYXWV9876U linked to mobile 8887776665.", {"entities": [(5, 15, "PAN"), (31, 41, "PHONE")]}),
]


