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
using ((select auth.uid()) = user_id);

create policy "User can insert insert their own transactions"
on transactions for insert
with check ((select auth.uid()) = user_id);

-- Add index to optimize foreign key performance
create index concurrently if not exists idx_transactions_user_id
  on transactions(user_id);

-- Create the budgets table
create table if not exists budgets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id),
  category text not null,
  budget_limit numeric not null,
  total_budget numeric not null,
  created_at timestamp with time zone default now()
);

-- Enable RLS for budgets
ALTER TABLE budgets ENABLE ROW LEVEL SECURITY;

-- Add RLS policies for budgets
create policy "User can view their own budgets"
on budgets for select
using ((select auth.uid()) = user_id);

create policy "User can insert their own budgets"
on budgets for insert
with check ((select auth.uid()) = user_id);

create policy "User can update their own budgets"
on budgets for update
using ((select auth.uid()) = user_id);

create policy "User can delete their own budgets"
on budgets for delete
using ((select auth.uid()) = user_id);

-- Add index to optimize foreign key performance
create index concurrently if not exists idx_budgets_user_id
  on budgets(user_id);