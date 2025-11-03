--Create the transactions table
create table if not exists transactions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id),
  transaction_date DATE,
  description text,
  transaction_amount numeric,
  category text,
  merchant text,
  account_name text,
  type text
);

-- Enable RLS
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

-- Add RLS policy
create policy "User can view their own transactions"
on transactions for select
using (auth.uid() = user_id);

create policy "User can insert insert their own transactions"
on transactions for insert
with check (auth.uid() = user_id);

-- Add index to optimize foreign key performance
create index concurrently if not exists idx_transactions_user_id
  on transactions(user_id);