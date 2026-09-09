extern crate proc_macro;
extern crate quote;
use proc_macro::*;

#[proc_macro_attribute]
pub fn autotest_annotate(attr: TokenStream, item: TokenStream) -> TokenStream {
    // let input = syn::parse_macro_input!(i as syn::ItemFn);
    let mut i: syn::Item = syn::parse(item).unwrap();
    let fn_item = match &mut i {
        syn::Item::Fn(fn_item) => fn_item,
        _ => panic!("expected fn"),
    };
    let fn_name = fn_item.sig.ident.to_string();
    let anno_name = attr.to_string();
    let anno = format!("!! liquid test annotation: {} -> {} !!", anno_name, fn_name);

    fn_item.block.stmts.insert(0, syn::parse_quote! { println!(#anno); });

    use quote::ToTokens;
    i.into_token_stream().into()
}
